import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Locale;

// Import COMSOL API libraries
import com.comsol.model.*;
import com.comsol.model.util.*;

/**
 * COMSOL Multiphysics verification driver for Stage 1 Hydraulic Solver.
 * 
 * Simulates 2D laminar Stokes/Navier-Stokes flow in an 8x8 tile chip:
 *   - Fixed single inlet at cell (0, 3) North face (inflow U0in = 0.0001 m/s)
 *   - Fixed 3 outlets at cells (7, 0), (7, 3), and (7, 7) South face (p = 0 Pa)
 * 
 * Usage:
 *   java -cp "<comsol plugins>/*;." ChipProcessor "<run_dir>"
 */
public class ChipProcessor {

    // --- DEFAULT GEOMETRY PARAMETERS (micrometers) ---
    private static double PITCH = 1200.0;
    private static double CHANNEL_WIDTH = 100.0;
    private static double L_STUB = 600.0;
    private static double INLET_FLOW_VELOCITY = 0.0001; // m/s
    private static int GRID_SIZE = 8;

    // --- OUTLET RATIOS ---
    private static double[] OUTLET_RATIOS = {1.0, 1.0, 1.0};

    // --- Boundary selections ---
    private static int[] inletBnds;
    private static int[] outlet1_Bnds;
    private static int[] outlet2_Bnds;
    private static int[] outlet3_Bnds;

    static class TileInfo {
        int row;
        int col;
        List<String> inletSides = new ArrayList<>();
        List<String> outletSides = new ArrayList<>();
        Map<String, Double> lengths = new HashMap<>();
    }

    private static List<TileInfo> tiles = new ArrayList<>();

    public static void main(String[] args) {
        String runDir = args.length > 0 ? args[0] : ".";
        Model model = null;
        try {
            ModelUtil.initStandalone(false);

            String ratioPath = runDir + File.separator + "ratios.txt";
            readRatiosFromFile(ratioPath);
            System.out.printf(Locale.US, "[ChipProcessor] Outlet Ratios: %.4f, %.4f, %.4f%n",
                    OUTLET_RATIOS[0], OUTLET_RATIOS[1], OUTLET_RATIOS[2]);

            String layoutPath = runDir + File.separator + "chip_layout.txt";
            if (!readChipLayoutFromFile(layoutPath)) {
                System.err.println("[ChipProcessor] Failed to read chip layout: " + layoutPath);
                System.exit(1);
            }
            System.out.printf(Locale.US, "[ChipProcessor] Loaded %d tiles from %s%n", tiles.size(), layoutPath);

            model = ModelUtil.create("ChipVerification");
            model.component().create("comp1", true);
            model.component("comp1").geom().create("geom1", 2);
            model.component("comp1").geom("geom1").lengthUnit("um");

            System.out.println("\n--- STARTING GEOMETRY BUILD ---");
            buildComsolGeometry(model);

            System.out.println("\n--- SETTING UP SIMULATION ---");
            assignBoundaries(model);
            setupPhysicsAndMaterials(model);
            configureBoundaryConditions(model);
            buildMesh(model);
            setupStudyAndSolver(model);

            System.out.println("\n--- RUNNING COMSOL SIMULATION ---");
            model.sol("sol1").runAll();
            System.out.println("[ChipProcessor] Simulation completed successfully.");

            double[] outletVelocities = getOutletVelocities(model);
            System.out.printf(Locale.US, "\n>>> Final Outlet Velocities [m/s]:%n");
            System.out.printf(Locale.US, "    Outlet 1 (7,0):      %.6f%n", outletVelocities[0]);
            System.out.printf(Locale.US, "    Outlet 2 (7,3):      %.6f%n", outletVelocities[1]);
            System.out.printf(Locale.US, "    Outlet 3 (7,7):      %.6f%n", outletVelocities[2]);

            // Save model file
            String modelFile = runDir + File.separator + "comsol_model.mph";
            System.out.println("Saving model to: " + modelFile);
            model.save(modelFile);

        } catch (Exception e) {
            System.err.println("[ChipProcessor] An exception occurred during the COMSOL run:");
            e.printStackTrace();
        } finally {
            if (model != null) {
                ModelUtil.disconnect();
            }
        }
        System.exit(0);
    }

    public static void buildComsolGeometry(Model model) {
        List<String> allTags = new ArrayList<>();
        double w = CHANNEL_WIDTH;
        double halfPitch = PITCH / 2.0;
        int tagCounter = 1;

        // 1. Build tile channels
        for (TileInfo tile : tiles) {
            double xC = (tile.col + 0.5) * PITCH;
            double yC = (GRID_SIZE - 1 - tile.row + 0.5) * PITCH;

            // Center junction square
            String cTag = "C_junc_" + tagCounter++;
            model.component("comp1").geom("geom1").create(cTag, "Rectangle");
            model.component("comp1").geom("geom1").feature(cTag).set("pos", new double[]{xC - w / 2.0, yC - w / 2.0});
            model.component("comp1").geom("geom1").feature(cTag).set("size", new double[]{w, w});
            allTags.add(cTag);

            // Active legs
            List<String> activeSides = new ArrayList<>(tile.inletSides);
            for (String s : tile.outletSides) {
                if (!activeSides.contains(s)) activeSides.add(s);
            }

            for (String s : activeSides) {
                double len = tile.lengths.getOrDefault(s, halfPitch);
                double extra = Math.max(0.0, len - halfPitch);

                if (s.equals("N")) {
                    if (extra <= 10.0) {
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg N: STRAIGHT (len=%.1f um)%n", tile.row, tile.col, halfPitch);
                        // Straight channel
                        String tag = "leg_N_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(tag, "Rectangle");
                        model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{xC - w / 2.0, yC});
                        model.component("comp1").geom("geom1").feature(tag).set("size", new double[]{w, halfPitch});
                        allTags.add(tag);
                    } else {
                        // Multi-pass serpentine channel (k = 1 or 2 loops to prevent overlapping while fitting high R)
                        int k = (extra > 1000.0) ? 2 : 1;
                        int nSegs = 2 * k + 1;
                        double segH = halfPitch / (double) nSegs;
                        double maxSpan = halfPitch - w;
                        double span = Math.min(extra / (2.0 * k), maxSpan);
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg N: SERPENTINE (len=%.1f um, extra=%.1f um, k=%d loops, span=%.1f um)%n",
                                tile.row, tile.col, len, extra, k, span);

                        for (int j = 0; j < k; j++) {
                            // Straight segment along junction line
                            String sA = "leg_N_sA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sA).set("pos", new double[]{xC - w / 2.0, yC + 2.0 * j * segH});
                            model.component("comp1").geom("geom1").feature(sA).set("size", new double[]{w, segH});
                            allTags.add(sA);

                            // Outward horizontal bend (+x)
                            String bA = "leg_N_bA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bA).set("pos", new double[]{xC - w / 2.0, yC + (2.0 * j + 1.0) * segH - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bA).set("size", new double[]{span + w, w});
                            allTags.add(bA);

                            // Detour vertical segment at +span
                            String sB = "leg_N_sB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sB).set("pos", new double[]{xC + span - w / 2.0, yC + (2.0 * j + 1.0) * segH});
                            model.component("comp1").geom("geom1").feature(sB).set("size", new double[]{w, segH});
                            allTags.add(sB);

                            // Inward horizontal bend (-x back to junction line)
                            String bB = "leg_N_bB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bB).set("pos", new double[]{xC - w / 2.0, yC + (2.0 * j + 2.0) * segH - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bB).set("size", new double[]{span + w, w});
                            allTags.add(bB);
                        }

                        // Final straight segment reaching cell boundary
                        String sFinal = "leg_N_sEnd_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(sFinal, "Rectangle");
                        model.component("comp1").geom("geom1").feature(sFinal).set("pos", new double[]{xC - w / 2.0, yC + 2.0 * k * segH});
                        model.component("comp1").geom("geom1").feature(sFinal).set("size", new double[]{w, segH});
                        allTags.add(sFinal);
                    }
                } else if (s.equals("S")) {
                    if (extra <= 10.0) {
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg S: STRAIGHT (len=%.1f um)%n", tile.row, tile.col, halfPitch);
                        String tag = "leg_S_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(tag, "Rectangle");
                        model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{xC - w / 2.0, yC - halfPitch});
                        model.component("comp1").geom("geom1").feature(tag).set("size", new double[]{w, halfPitch});
                        allTags.add(tag);
                    } else {
                        int k = (extra > 1000.0) ? 2 : 1;
                        int nSegs = 2 * k + 1;
                        double segH = halfPitch / (double) nSegs;
                        double maxSpan = halfPitch - w;
                        double span = Math.min(extra / (2.0 * k), maxSpan);
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg S: SERPENTINE (len=%.1f um, extra=%.1f um, k=%d loops, span=%.1f um)%n",
                                tile.row, tile.col, len, extra, k, span);

                        for (int j = 0; j < k; j++) {
                            // Downward segment along junction line
                            String sA = "leg_S_sA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sA).set("pos", new double[]{xC - w / 2.0, yC - (2.0 * j + 1.0) * segH});
                            model.component("comp1").geom("geom1").feature(sA).set("size", new double[]{w, segH});
                            allTags.add(sA);

                            // Outward horizontal bend (-x)
                            String bA = "leg_S_bA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bA).set("pos", new double[]{xC - span - w / 2.0, yC - (2.0 * j + 1.0) * segH - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bA).set("size", new double[]{span + w, w});
                            allTags.add(bA);

                            // Detour downward segment at -span
                            String sB = "leg_S_sB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sB).set("pos", new double[]{xC - span - w / 2.0, yC - (2.0 * j + 2.0) * segH});
                            model.component("comp1").geom("geom1").feature(sB).set("size", new double[]{w, segH});
                            allTags.add(sB);

                            // Inward horizontal bend (+x back to junction line)
                            String bB = "leg_S_bB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bB).set("pos", new double[]{xC - span - w / 2.0, yC - (2.0 * j + 2.0) * segH - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bB).set("size", new double[]{span + w, w});
                            allTags.add(bB);
                        }

                        // Final downward segment reaching cell boundary
                        String sFinal = "leg_S_sEnd_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(sFinal, "Rectangle");
                        model.component("comp1").geom("geom1").feature(sFinal).set("pos", new double[]{xC - w / 2.0, yC - (2.0 * k + 1.0) * segH});
                        model.component("comp1").geom("geom1").feature(sFinal).set("size", new double[]{w, segH});
                        allTags.add(sFinal);
                    }
                } else if (s.equals("W")) {
                    if (extra <= 10.0) {
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg W: STRAIGHT (len=%.1f um)%n", tile.row, tile.col, halfPitch);
                        String tag = "leg_W_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(tag, "Rectangle");
                        model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{xC - halfPitch, yC - w / 2.0});
                        model.component("comp1").geom("geom1").feature(tag).set("size", new double[]{halfPitch, w});
                        allTags.add(tag);
                    } else {
                        int k = (extra > 1000.0) ? 2 : 1;
                        int nSegs = 2 * k + 1;
                        double segW = halfPitch / (double) nSegs;
                        double maxSpan = halfPitch - w;
                        double span = Math.min(extra / (2.0 * k), maxSpan);
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg W: SERPENTINE (len=%.1f um, extra=%.1f um, k=%d loops, span=%.1f um)%n",
                                tile.row, tile.col, len, extra, k, span);

                        for (int j = 0; j < k; j++) {
                            // Leftward segment along junction line
                            String sA = "leg_W_sA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sA).set("pos", new double[]{xC - (2.0 * j + 1.0) * segW, yC - w / 2.0});
                            model.component("comp1").geom("geom1").feature(sA).set("size", new double[]{segW, w});
                            allTags.add(sA);

                            // Outward vertical bend (+y)
                            String bA = "leg_W_bA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bA).set("pos", new double[]{xC - (2.0 * j + 1.0) * segW - w / 2.0, yC - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bA).set("size", new double[]{w, span + w});
                            allTags.add(bA);

                            // Detour leftward segment at +span
                            String sB = "leg_W_sB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sB).set("pos", new double[]{xC - (2.0 * j + 2.0) * segW, yC + span - w / 2.0});
                            model.component("comp1").geom("geom1").feature(sB).set("size", new double[]{segW, w});
                            allTags.add(sB);

                            // Inward vertical bend (-y back to junction line)
                            String bB = "leg_W_bB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bB).set("pos", new double[]{xC - (2.0 * j + 2.0) * segW - w / 2.0, yC - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bB).set("size", new double[]{w, span + w});
                            allTags.add(bB);
                        }

                        // Final leftward segment reaching cell boundary
                        String sFinal = "leg_W_sEnd_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(sFinal, "Rectangle");
                        model.component("comp1").geom("geom1").feature(sFinal).set("pos", new double[]{xC - (2.0 * k + 1.0) * segW, yC - w / 2.0});
                        model.component("comp1").geom("geom1").feature(sFinal).set("size", new double[]{segW, w});
                        allTags.add(sFinal);
                    }
                } else if (s.equals("E")) {
                    if (extra <= 10.0) {
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg E: STRAIGHT (len=%.1f um)%n", tile.row, tile.col, halfPitch);
                        String tag = "leg_E_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(tag, "Rectangle");
                        model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{xC, yC - w / 2.0});
                        model.component("comp1").geom("geom1").feature(tag).set("size", new double[]{halfPitch, w});
                        allTags.add(tag);
                    } else {
                        int k = (extra > 1000.0) ? 2 : 1;
                        int nSegs = 2 * k + 1;
                        double segW = halfPitch / (double) nSegs;
                        double maxSpan = halfPitch - w;
                        double span = Math.min(extra / (2.0 * k), maxSpan);
                        System.out.printf(Locale.US, "  [CAD] Tile (%d,%d) leg E: SERPENTINE (len=%.1f um, extra=%.1f um, k=%d loops, span=%.1f um)%n",
                                tile.row, tile.col, len, extra, k, span);

                        for (int j = 0; j < k; j++) {
                            // Rightward segment along junction line
                            String sA = "leg_E_sA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sA).set("pos", new double[]{xC + 2.0 * j * segW, yC - w / 2.0});
                            model.component("comp1").geom("geom1").feature(sA).set("size", new double[]{segW, w});
                            allTags.add(sA);

                            // Outward vertical bend (-y)
                            String bA = "leg_E_bA_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bA, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bA).set("pos", new double[]{xC + (2.0 * j + 1.0) * segW - w / 2.0, yC - span - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bA).set("size", new double[]{w, span + w});
                            allTags.add(bA);

                            // Detour rightward segment at -span
                            String sB = "leg_E_sB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(sB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(sB).set("pos", new double[]{xC + (2.0 * j + 1.0) * segW, yC - span - w / 2.0});
                            model.component("comp1").geom("geom1").feature(sB).set("size", new double[]{segW, w});
                            allTags.add(sB);

                            // Inward vertical bend (+y back to junction line)
                            String bB = "leg_E_bB_" + j + "_" + tagCounter++;
                            model.component("comp1").geom("geom1").create(bB, "Rectangle");
                            model.component("comp1").geom("geom1").feature(bB).set("pos", new double[]{xC + (2.0 * j + 2.0) * segW - w / 2.0, yC - span - w / 2.0});
                            model.component("comp1").geom("geom1").feature(bB).set("size", new double[]{w, span + w});
                            allTags.add(bB);
                        }

                        // Final rightward segment reaching cell boundary
                        String sFinal = "leg_E_sEnd_" + tagCounter++;
                        model.component("comp1").geom("geom1").create(sFinal, "Rectangle");
                        model.component("comp1").geom("geom1").feature(sFinal).set("pos", new double[]{xC + 2.0 * k * segW, yC - w / 2.0});
                        model.component("comp1").geom("geom1").feature(sFinal).set("size", new double[]{segW, w});
                        allTags.add(sFinal);
                    }
                }
            }
        }

        // 2. Inlet stub at cell (0, 3) North face
        double xInlet = 3.5 * PITCH;
        double yInlet = GRID_SIZE * PITCH;
        model.component("comp1").geom("geom1").create("inlet_stub", "Rectangle");
        model.component("comp1").geom("geom1").feature("inlet_stub").set("pos", new double[]{xInlet - w / 2.0, yInlet});
        model.component("comp1").geom("geom1").feature("inlet_stub").set("size", new double[]{w, L_STUB});
        allTags.add("inlet_stub");

        // 3. Outlet stubs at cells (7, 0), (7, 3), (7, 7) South face
        double[] outCols = {0.5 * PITCH, 3.5 * PITCH, 7.5 * PITCH};
        for (int i = 0; i < 3; i++) {
            double outLen = L_STUB * OUTLET_RATIOS[i];
            String oTag = "outlet_stub_" + (i + 1);
            model.component("comp1").geom("geom1").create(oTag, "Rectangle");
            model.component("comp1").geom("geom1").feature(oTag).set("pos", new double[]{outCols[i] - w / 2.0, -outLen});
            model.component("comp1").geom("geom1").feature(oTag).set("size", new double[]{w, outLen});
            allTags.add(oTag);
        }

        // 4. Geometry union
        if (!allTags.isEmpty()) {
            model.component("comp1").geom("geom1").create("u1", "Union");
            model.component("comp1").geom("geom1").feature("u1").selection("input").set(allTags.toArray(new String[0]));
            model.component("comp1").geom("geom1").feature("u1").set("intbnd", false);
            model.component("comp1").geom("geom1").run();
        }
    }

    public static void assignBoundaries(Model model) {
        System.out.println("Selecting boundary ports...");
        double ballR = Math.min(15.0, 0.25 * CHANNEL_WIDTH);

        // Inlet boundary at (3.5 * PITCH, GRID_SIZE * PITCH + L_STUB)
        double xInlet = 3.5 * PITCH;
        double yInlet = GRID_SIZE * PITCH + L_STUB;
        model.component("comp1").selection().create("sel_in", "Ball");
        model.component("comp1").selection("sel_in").set("entitydim", 1);
        model.component("comp1").selection("sel_in").set("posx", xInlet);
        model.component("comp1").selection("sel_in").set("posy", yInlet);
        model.component("comp1").selection("sel_in").set("r", ballR);
        inletBnds = model.component("comp1").selection("sel_in").entities();

        // Outlet 1 boundary at (0.5 * PITCH, -L_STUB * OUTLET_RATIOS[0])
        double out1Y = -L_STUB * OUTLET_RATIOS[0];
        model.component("comp1").selection().create("sel_out1", "Ball");
        model.component("comp1").selection("sel_out1").set("entitydim", 1);
        model.component("comp1").selection("sel_out1").set("posx", 0.5 * PITCH);
        model.component("comp1").selection("sel_out1").set("posy", out1Y);
        model.component("comp1").selection("sel_out1").set("r", ballR);
        outlet1_Bnds = model.component("comp1").selection("sel_out1").entities();

        // Outlet 2 boundary at (3.5 * PITCH, -L_STUB * OUTLET_RATIOS[1])
        double out2Y = -L_STUB * OUTLET_RATIOS[1];
        model.component("comp1").selection().create("sel_out2", "Ball");
        model.component("comp1").selection("sel_out2").set("entitydim", 1);
        model.component("comp1").selection("sel_out2").set("posx", 3.5 * PITCH);
        model.component("comp1").selection("sel_out2").set("posy", out2Y);
        model.component("comp1").selection("sel_out2").set("r", ballR);
        outlet2_Bnds = model.component("comp1").selection("sel_out2").entities();

        // Outlet 3 boundary at (7.5 * PITCH, -L_STUB * OUTLET_RATIOS[2])
        double out3Y = -L_STUB * OUTLET_RATIOS[2];
        model.component("comp1").selection().create("sel_out3", "Ball");
        model.component("comp1").selection("sel_out3").set("entitydim", 1);
        model.component("comp1").selection("sel_out3").set("posx", 7.5 * PITCH);
        model.component("comp1").selection("sel_out3").set("posy", out3Y);
        model.component("comp1").selection("sel_out3").set("r", ballR);
        outlet3_Bnds = model.component("comp1").selection("sel_out3").entities();
    }

    public static void setupPhysicsAndMaterials(Model model) {
        System.out.println("Configuring Laminar Flow (spf) physics and water properties...");
        model.component("comp1").physics().create("spf", "LaminarFlow", "geom1");
        model.component("comp1").material().create("mat1", "Common");
        model.component("comp1").material("mat1").label("Water");
        model.component("comp1").material("mat1").propertyGroup("def").set("dynamicviscosity", "1.01e-3 [Pa*s]");
        model.component("comp1").material("mat1").propertyGroup("def").set("density", "998.2 [kg/m^3]");
    }

    public static void configureBoundaryConditions(Model model) {
        System.out.println("Applying boundary conditions...");
        if (inletBnds != null && inletBnds.length > 0) {
            model.component("comp1").physics("spf").create("in1_spf", "InletBoundary", 1);
            model.component("comp1").physics("spf").feature("in1_spf").set("U0in", INLET_FLOW_VELOCITY + " [m/s]");
            model.component("comp1").physics("spf").feature("in1_spf").selection().set(inletBnds);
        } else {
            System.err.println("WARNING: Inlet boundary selection is empty!");
        }

        int[][] outs = {outlet1_Bnds, outlet2_Bnds, outlet3_Bnds};
        for (int i = 0; i < 3; i++) {
            if (outs[i] != null && outs[i].length > 0) {
                String tag = "out" + (i + 1) + "_spf";
                model.component("comp1").physics("spf").create(tag, "OutletBoundary", 1);
                model.component("comp1").physics("spf").feature(tag).set("p0", "0 [Pa]");
                model.component("comp1").physics("spf").feature(tag).selection().set(outs[i]);
            } else {
                System.err.println("WARNING: Outlet " + (i + 1) + " boundary selection is empty!");
            }
        }
    }

    public static void buildMesh(Model model) {
        System.out.println("Building mesh...");
        model.component("comp1").mesh().create("mesh1");
        model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
        model.component("comp1").mesh("mesh1").feature("size").set("hmax", 50.0);
        model.component("comp1").mesh("mesh1").feature("size").set("hmin", 10.0);
        model.component("comp1").mesh("mesh1").run();
    }

    public static void setupStudyAndSolver(Model model) {
        System.out.println("Setting up Stationary study and PARDISO solver...");
        model.study().create("std1");
        model.study("std1").create("stat", "Stationary");
        model.study("std1").feature("stat").set("activate", new String[]{"spf", "on"});
        model.sol().create("sol1");
        model.sol("sol1").study("std1");
        model.sol("sol1").create("st1", "StudyStep");
        model.sol("sol1").feature("st1").set("studystep", "stat");
        model.sol("sol1").create("v1", "Variables");
        model.sol("sol1").create("s1", "Stationary");
        model.sol("sol1").feature("s1").feature().create("fc1", "FullyCoupled");
        model.sol("sol1").feature("s1").feature("fc1").set("maxiter", 100);
        model.sol("sol1").feature("s1").create("d1", "Direct");
        model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
        model.sol("sol1").attach("std1");
    }

    public static double[] getOutletVelocities(Model model) {
        double[] velocities = new double[3];
        String[] tags = {"av_vel_out1", "av_vel_out2", "av_vel_out3"};
        int[][] bnds = {outlet1_Bnds, outlet2_Bnds, outlet3_Bnds};

        for (int i = 0; i < 3; i++) {
            if (bnds[i] != null && bnds[i].length > 0) {
                try {
                    model.result().numerical().create(tags[i], "AvLine");
                    model.result().numerical(tags[i]).setIndex("expr", "spf.U", 0);
                    model.result().numerical(tags[i]).selection().set(bnds[i]);
                    double[][][] table = model.result().numerical(tags[i]).computeResult();
                    velocities[i] = table[0][0][0];
                    model.result().numerical().remove(tags[i]);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        return velocities;
    }

    public static void readRatiosFromFile(String filename) {
        try (Scanner scanner = new Scanner(new File(filename))) {
            scanner.useLocale(Locale.US);
            if (scanner.hasNextDouble()) OUTLET_RATIOS[0] = scanner.nextDouble();
            if (scanner.hasNextDouble()) OUTLET_RATIOS[1] = scanner.nextDouble();
            if (scanner.hasNextDouble()) OUTLET_RATIOS[2] = scanner.nextDouble();
        } catch (FileNotFoundException e) {
            System.out.println("Warning: ratios.txt not found. Using default {1.0, 1.0, 1.0}.");
        } catch (Exception e) {
            System.out.println("Warning: Error reading ratios.txt: " + e.getMessage());
        }
    }

    public static boolean readChipLayoutFromFile(String filename) {
        tiles.clear();
        try (Scanner sc = new Scanner(new File(filename))) {
            sc.useLocale(Locale.US);
            while (sc.hasNextLine()) {
                String line = sc.nextLine().trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+");
                String cmd = parts[0].toUpperCase();

                if (cmd.equals("GRID_SIZE")) {
                    GRID_SIZE = Integer.parseInt(parts[1]);
                } else if (cmd.equals("PITCH")) {
                    PITCH = Double.parseDouble(parts[1]);
                } else if (cmd.equals("CHANNEL_WIDTH")) {
                    CHANNEL_WIDTH = Double.parseDouble(parts[1]);
                } else if (cmd.equals("L_STUB")) {
                    L_STUB = Double.parseDouble(parts[1]);
                } else if (cmd.equals("INLET_FLOW")) {
                    INLET_FLOW_VELOCITY = Double.parseDouble(parts[1]);
                } else if (cmd.equals("TILE")) {
                    TileInfo t = new TileInfo();
                    t.row = Integer.parseInt(parts[1]);
                    t.col = Integer.parseInt(parts[2]);
                    if (!parts[3].equals("none")) {
                        for (String s : parts[3].split(";")) if (!s.isEmpty()) t.inletSides.add(s);
                    }
                    if (!parts[4].equals("none")) {
                        for (String s : parts[4].split(";")) if (!s.isEmpty()) t.outletSides.add(s);
                    }
                    t.lengths.put("N", Double.parseDouble(parts[5]));
                    t.lengths.put("S", Double.parseDouble(parts[6]));
                    t.lengths.put("W", Double.parseDouble(parts[7]));
                    t.lengths.put("E", Double.parseDouble(parts[8]));
                    tiles.add(t);
                    System.out.printf(Locale.US, "  [Layout] Tile (%d,%d) in=[%s] out=[%s] | lens: N=%.1f S=%.1f W=%.1f E=%.1f um%n",
                            t.row, t.col, parts[3], parts[4], t.lengths.get("N"), t.lengths.get("S"), t.lengths.get("W"), t.lengths.get("E"));
                }
            }
            System.out.printf(Locale.US, "[ChipProcessor] Successfully parsed %d tiles from layout.%n", tiles.size());
            return !tiles.isEmpty();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}


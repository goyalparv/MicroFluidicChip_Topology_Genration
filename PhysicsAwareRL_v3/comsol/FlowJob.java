import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// COMSOL API
import com.comsol.model.*;
import com.comsol.model.util.*;

/**
 * Generic 2D laminar-flow job for multi-port microfluidic tiles in PhysicsAwareRL_v2.
 *
 *   java -cp "<comsol plugins>/*;<bin>" FlowJob job.txt result.txt
 */
public class FlowJob {

    static Map<String, String> scalar = new HashMap<String, String>();
    static List<double[]> rects = new ArrayList<double[]>();
    static List<String[]> ports = new ArrayList<String[]>();   // {kind, x, y, expr}

    public static void main(String[] args) {
        String jobFile = args.length > 0 ? args[0] : "job.txt";
        String resFile = args.length > 1 ? args[1] : "result.txt";
        StringBuilder out = new StringBuilder();
        Model model = null;
        try {
            readJob(jobFile);
            double w = Double.parseDouble(scalar.get("pixel_um"));
            double thick = Double.parseDouble(scalar.get("thickness_um"));
            double mu = Double.parseDouble(scalar.get("viscosity"));
            double rho = Double.parseDouble(scalar.get("density"));
            double uin = Double.parseDouble(scalar.get("u_in"));
            double hmax = Double.parseDouble(scalar.get("mesh_hmax_um"));
            double hmin = Double.parseDouble(scalar.get("mesh_hmin_um"));
            int nRows = Integer.parseInt(scalar.get("n_rows"));
            boolean saveMph = "1".equals(scalar.get("save_mph"));

            ModelUtil.initStandalone(false);
            model = ModelUtil.create("FlowJob");
            model.component().create("comp1", true);
            model.component("comp1").geom().create("geom1", 2);
            model.component("comp1").geom("geom1").lengthUnit("um");

            buildGeometry(model, w, nRows);
            int[][] bnds = selectPorts(model, w);
            boolean thicknessApplied = setupPhysics(model, mu, rho, thick, uin, bnds);
            buildMeshAndSolve(model, hmax, hmin);
            collectResults(model, bnds, thicknessApplied, out);

            if (saveMph) {
                model.save(new File(".").getAbsolutePath() + File.separator + "model.mph");
            }
        } catch (Exception e) {
            e.printStackTrace();
            out.setLength(0);
            out.append("status error\n");
            out.append("message ").append(String.valueOf(e.getMessage()).replace('\n', ' ')).append('\n');
        } finally {
            try {
                write(resFile, out.toString());
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
            if (model != null) {
                ModelUtil.disconnect();
            }
        }
        System.exit(0);
    }

    static void buildGeometry(Model model, double w, int nRows) {
        List<String> tags = new ArrayList<String>();
        for (int i = 0; i < rects.size(); i++) {
            double[] r = rects.get(i);                  // row0, col0, nRows, nCols (raster row 0 = top)
            String tag = "R" + i;
            double x = r[1] * w;
            double y = (nRows - r[0] - r[2]) * w;
            model.component("comp1").geom("geom1").create(tag, "Rectangle");
            model.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{x, y});
            model.component("comp1").geom("geom1").feature(tag).set("size", new double[]{r[3] * w, r[2] * w});
            tags.add(tag);
        }
        model.component("comp1").geom("geom1").create("u1", "Union");
        model.component("comp1").geom("geom1").feature("u1").selection("input").set(tags.toArray(new String[0]));
        model.component("comp1").geom("geom1").feature("u1").set("intbnd", false);
        model.component("comp1").geom("geom1").run();
    }

    static int[][] selectPorts(Model model, double w) {
        double ballR = Math.min(10.0, 0.2 * w);
        int[][] bnds = new int[ports.size()][];
        for (int i = 0; i < ports.size(); i++) {
            String[] p = ports.get(i);
            String sel = "sel_p" + i;
            model.component("comp1").selection().create(sel, "Ball");
            model.component("comp1").selection(sel).set("entitydim", 1);
            model.component("comp1").selection(sel).set("posx", Double.parseDouble(p[1]));
            model.component("comp1").selection(sel).set("posy", Double.parseDouble(p[2]));
            model.component("comp1").selection(sel).set("r", ballR);
            bnds[i] = model.component("comp1").selection(sel).entities();
            if (bnds[i] == null || bnds[i].length == 0) {
                throw new RuntimeException("port " + i + " (" + p[0] + ") selected no boundary at ("
                        + p[1] + ", " + p[2] + ") um");
            }
        }
        return bnds;
    }

    static boolean setupPhysics(Model model, double mu, double rho, double thick, double uin, int[][] bnds) {
        model.component("comp1").physics().create("spf", "LaminarFlow", "geom1");
        model.component("comp1").material().create("mat1", "Common");
        model.component("comp1").material("mat1").label("Fluid");
        model.component("comp1").material("mat1").propertyGroup("def").set("dynamicviscosity", mu + " [Pa*s]");
        model.component("comp1").material("mat1").propertyGroup("def").set("density", rho + " [kg/m^3]");

        boolean thicknessApplied = true;
        if (thick > 0) {
            thicknessApplied = false;
            try {
                model.component("comp1").physics("spf").prop("d").set("d", thick + "[um]");
                thicknessApplied = true;
            } catch (Exception e) {
                System.out.println("could not set the out-of-plane thickness: " + e.getMessage());
            }
        }
        for (int i = 0; i < ports.size(); i++) {
            if (ports.get(i)[0].equals("inlet")) {
                model.component("comp1").physics("spf").create("inlet", "InletBoundary", 1);
                model.component("comp1").physics("spf").feature("inlet").set("U0in", uin + "[m/s]");
                model.component("comp1").physics("spf").feature("inlet").selection().set(bnds[i]);
            } else {
                String tag = "outlet" + i;
                model.component("comp1").physics("spf").create(tag, "OutletBoundary", 1);
                model.component("comp1").physics("spf").feature(tag).set("p0", "0[Pa]");
                model.component("comp1").physics("spf").feature(tag).selection().set(bnds[i]);
            }
        }
        return thicknessApplied;
    }

    static void buildMeshAndSolve(Model model, double hmax, double hmin) {
        model.component("comp1").mesh().create("mesh1");
        model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
        model.component("comp1").mesh("mesh1").feature("size").set("hmax", hmax);
        model.component("comp1").mesh("mesh1").feature("size").set("hmin", hmin);
        model.component("comp1").mesh("mesh1").run();

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
        model.sol("sol1").runAll();
    }

    static void collectResults(Model model, int[][] bnds, boolean thicknessApplied, StringBuilder out) {
        double pIn = Double.NaN;
        double inAvg = Double.NaN;
        StringBuilder outs = new StringBuilder();
        int k = 0;
        for (int i = 0; i < ports.size(); i++) {
            String[] p = ports.get(i);
            if (p[0].equals("inlet")) {
                pIn = avg(model, "av_p", "p", bnds[i]);
                inAvg = avg(model, "av_in", p[3], bnds[i]);
            } else {
                double v = avg(model, "av_out" + i, p[3], bnds[i]);
                outs.append("out_avg ").append(k).append(' ').append(v).append('\n');
                k++;
            }
        }
        out.append("status ok\n");
        out.append("p_in ").append(pIn).append('\n');
        out.append("in_avg ").append(inAvg).append('\n');
        out.append("thickness_applied ").append(thicknessApplied ? 1 : 0).append('\n');
        out.append(outs);
    }

    static double avg(Model model, String tag, String expr, int[] bnd) {
        model.result().numerical().create(tag, "AvLine");
        model.result().numerical(tag).setIndex("expr", expr, 0);
        model.result().numerical(tag).selection().set(bnd);
        double[][][] table = model.result().numerical(tag).computeResult();
        double v = table[0][0][0];
        model.result().numerical().remove(tag);
        return v;
    }

    static void readJob(String file) throws IOException {
        for (String line : Files.readAllLines(new File(file).toPath(), StandardCharsets.UTF_8)) {
            String[] t = line.trim().split("\\s+");
            if (t.length < 2) {
                continue;
            }
            if (t[0].equals("rect")) {
                rects.add(new double[]{Double.parseDouble(t[1]), Double.parseDouble(t[2]),
                        Double.parseDouble(t[3]), Double.parseDouble(t[4])});
            } else if (t[0].equals("inlet") || t[0].equals("outlet")) {
                ports.add(new String[]{t[0], t[1], t[2], t[3]});
            } else {
                scalar.put(t[0], t[1]);
            }
        }
    }

    static void write(String file, String text) throws IOException {
        PrintWriter pw = new PrintWriter(new File(file), "UTF-8");
        pw.print(text);
        pw.close();
    }
}


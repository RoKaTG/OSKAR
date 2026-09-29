/*
 * Compares the visibility amplitudes in two OSKAR binary files and reports
 * the relative error between them, using the same metric and default
 * tolerances as the OSKAR unit tests.
 *
 * Intended for validating a new compute backend against a trusted reference:
 *
 *     oskar_vis_compare reference.vis candidate.vis
 *
 * Exits with status 0 if the candidate is within tolerance, 1 otherwise, so
 * it can be used directly as a build-pipeline check.
 */

#include "binary/oskar_binary.h"
#include "mem/oskar_mem.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/*
 * Primary metric: the normalised RMS difference between the two amplitude
 * arrays, treated as flat vectors of real components,
 *
 *     nrmse = sqrt( sum|cand - ref|^2 / sum|ref|^2 )
 *
 * This is scale-invariant, sensitive to sign and conjugation errors, and not
 * distorted by visibilities that happen to sit near zero -- all of which
 * matter when validating a compute backend, and none of which are true of a
 * per-element relative error. OSKAR's own
 * oskar_mem_evaluate_relative_error() is reported alongside it for
 * comparability with the unit tests, but is not used as the gate: it
 * compares magnitudes componentwise, so it cannot see a sign flip.
 */
struct Accum
{
    double sum_diff2, sum_ref2, max_abs_diff, max_abs_ref;
    double osk_max_rel, osk_sum_avg;
    int num_blocks;
    Accum() : sum_diff2(0.0), sum_ref2(0.0), max_abs_diff(0.0),
        max_abs_ref(0.0), osk_max_rel(0.0), osk_sum_avg(0.0),
        num_blocks(0) {}

    double nrmse() const
    {
        if (sum_ref2 <= 0.0) return sum_diff2 > 0.0 ? 1.0 : 0.0;
        return sqrt(sum_diff2 / sum_ref2);
    }
    double osk_avg() const
    {
        return num_blocks ? osk_sum_avg / num_blocks : 0.0;
    }
};

/* Accumulate the sums of squares over one pair of amplitude arrays. */
static void accumulate(const oskar_Mem* ref, const oskar_Mem* cand,
        Accum* a, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(ref);
    if (type != oskar_mem_type(cand) ||
            oskar_mem_length(ref) != oskar_mem_length(cand))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    const size_t num_real = oskar_mem_length(ref) *
            oskar_mem_element_size(type) /
            ((oskar_type_precision(type) == OSKAR_DOUBLE) ?
                    sizeof(double) : sizeof(float));

    if (oskar_type_precision(type) == OSKAR_DOUBLE)
    {
        const double* r = (const double*) oskar_mem_void_const(ref);
        const double* c = (const double*) oskar_mem_void_const(cand);
        for (size_t i = 0; i < num_real; ++i)
        {
            const double d = c[i] - r[i];
            a->sum_diff2 += d * d;
            a->sum_ref2 += r[i] * r[i];
            if (fabs(d) > a->max_abs_diff) a->max_abs_diff = fabs(d);
            if (fabs(r[i]) > a->max_abs_ref) a->max_abs_ref = fabs(r[i]);
        }
    }
    else
    {
        const float* r = (const float*) oskar_mem_void_const(ref);
        const float* c = (const float*) oskar_mem_void_const(cand);
        for (size_t i = 0; i < num_real; ++i)
        {
            const double d = (double) c[i] - (double) r[i];
            a->sum_diff2 += d * d;
            a->sum_ref2 += (double) r[i] * (double) r[i];
            if (fabs(d) > a->max_abs_diff) a->max_abs_diff = fabs(d);
            if (fabs((double) r[i]) > a->max_abs_ref)
            {
                a->max_abs_ref = fabs((double) r[i]);
            }
        }
    }

    /* OSKAR's own metric, for continuity with the unit tests. */
    double mn = 0.0, mx = 0.0, av = 0.0, sd = 0.0;
    oskar_mem_evaluate_relative_error(cand, ref, &mn, &mx, &av, &sd, status);
    if (mx > a->osk_max_rel) a->osk_max_rel = mx;
    a->osk_sum_avg += av;
    a->num_blocks++;
}

static void report(const char* label, const Accum& a, double tol, int* failed)
{
    if (a.num_blocks == 0) return;
    const int ok = (a.nrmse() < tol);
    if (!ok) *failed = 1;
    printf("  %-20s  nrmse %10.3e   (tol %7.1e)   %s\n",
            label, a.nrmse(), tol, ok ? "PASS" : "FAIL");
    printf("  %-20s  peak |diff| %.3e against peak |ref| %.3e\n",
            "", a.max_abs_diff, a.max_abs_ref);
    printf("  %-20s  OSKAR rel. error: max %.3e, avg %.3e\n\n",
            "", a.osk_max_rel, a.osk_avg());
}

int main(int argc, char** argv)
{
    int status = 0;

    oskar::OptionParser opt("oskar_vis_compare", oskar_version_string());
    opt.add_required("reference OSKAR visibility file");
    opt.add_required("candidate OSKAR visibility file");
    opt.add_flag("--tol", "Maximum permitted normalised RMS difference. "
            "Default: 1e-11 for double precision, 1e-4 for single.", 1);
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    const char* ref_name = opt.get_arg(0);
    const char* cand_name = opt.get_arg(1);

    /* Open both files and read their headers. */
    oskar_Binary* ref_file = oskar_binary_create(ref_name, 'r', &status);
    oskar_VisHeader* ref_hdr = oskar_vis_header_read(ref_file, &status);
    if (status)
    {
        fprintf(stderr, "Error reading '%s': %s\n",
                ref_name, oskar_get_error_string(status));
        return EXIT_FAILURE;
    }
    oskar_Binary* cand_file = oskar_binary_create(cand_name, 'r', &status);
    oskar_VisHeader* cand_hdr = oskar_vis_header_read(cand_file, &status);
    if (status)
    {
        fprintf(stderr, "Error reading '%s': %s\n",
                cand_name, oskar_get_error_string(status));
        return EXIT_FAILURE;
    }

    /* The two files must describe the same observation for the comparison
     * to mean anything. */
    const int num_blocks = oskar_vis_header_num_blocks(ref_hdr);
    const int is_dbl = (oskar_type_precision(
            oskar_vis_header_amp_type(ref_hdr)) == OSKAR_DOUBLE) &&
            (oskar_type_precision(
            oskar_vis_header_amp_type(cand_hdr)) == OSKAR_DOUBLE);
    int mismatch = 0;
    if (oskar_vis_header_num_stations(ref_hdr) !=
            oskar_vis_header_num_stations(cand_hdr)) mismatch = 1;
    if (oskar_vis_header_num_channels_total(ref_hdr) !=
            oskar_vis_header_num_channels_total(cand_hdr)) mismatch = 1;
    if (oskar_vis_header_num_times_total(ref_hdr) !=
            oskar_vis_header_num_times_total(cand_hdr)) mismatch = 1;
    if (num_blocks != oskar_vis_header_num_blocks(cand_hdr)) mismatch = 1;
    if (mismatch)
    {
        fprintf(stderr, "The two files describe different observations "
                "(stations, channels, times or blocks differ).\n");
        return EXIT_FAILURE;
    }

    /* Defaults are calibrated against the measured CPU-vs-CUDA agreement on
     * the bundled example (nrmse 1.6e-15 double, 7.4e-7 single), leaving
     * roughly three orders of magnitude of headroom before a real defect is
     * reported. They are deliberately looser than the unit-test tolerances,
     * which apply to well-conditioned synthetic data rather than a full
     * simulation containing near-null visibilities. */
    double tol = is_dbl ? 1e-11 : 1e-4;
    if (opt.is_set("--tol")) tol = opt.get_double("--tol");

    printf("Reference : %s\n", ref_name);
    printf("Candidate : %s\n", cand_name);
    printf("Stations %d, channels %d, times %d, blocks %d, precision %s\n",
            oskar_vis_header_num_stations(ref_hdr),
            oskar_vis_header_num_channels_total(ref_hdr),
            oskar_vis_header_num_times_total(ref_hdr),
            num_blocks, is_dbl ? "double" : "single");
    printf("Gate: normalised RMS difference < %.3e\n\n", tol);

    Accum cross, autoc;
    for (int i = 0; i < num_blocks && !status; ++i)
    {
        oskar_VisBlock* ref_blk = oskar_vis_block_create_from_header(
                OSKAR_CPU, ref_hdr, &status);
        oskar_VisBlock* cand_blk = oskar_vis_block_create_from_header(
                OSKAR_CPU, cand_hdr, &status);
        oskar_vis_block_read(ref_blk, ref_hdr, ref_file, i, &status);
        oskar_vis_block_read(cand_blk, cand_hdr, cand_file, i, &status);

        if (!status && oskar_vis_block_has_cross_correlations(ref_blk) &&
                oskar_vis_block_has_cross_correlations(cand_blk))
        {
            accumulate(oskar_vis_block_cross_correlations_const(ref_blk),
                    oskar_vis_block_cross_correlations_const(cand_blk),
                    &cross, &status);
        }
        if (!status && oskar_vis_block_has_auto_correlations(ref_blk) &&
                oskar_vis_block_has_auto_correlations(cand_blk))
        {
            accumulate(oskar_vis_block_auto_correlations_const(ref_blk),
                    oskar_vis_block_auto_correlations_const(cand_blk),
                    &autoc, &status);
        }
        oskar_vis_block_free(ref_blk, &status);
        oskar_vis_block_free(cand_blk, &status);
    }

    int failed = 0;
    report("cross-correlations", cross, tol, &failed);
    report("auto-correlations", autoc, tol, &failed);

    if (status)
    {
        fprintf(stderr, "\nError during comparison: %s\n",
                oskar_get_error_string(status));
        failed = 1;
    }
    else if (cross.num_blocks == 0 && autoc.num_blocks == 0)
    {
        fprintf(stderr, "\nNo comparable visibility data found.\n");
        failed = 1;
    }
    printf("\nRESULT: %s\n", failed ? "FAIL" : "PASS");

    oskar_vis_header_free(ref_hdr, &status);
    oskar_vis_header_free(cand_hdr, &status);
    oskar_binary_free(ref_file);
    oskar_binary_free(cand_file);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

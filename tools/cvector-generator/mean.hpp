#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <math.h>

namespace mean {

struct layer_report {
    int n_layers;
    std::vector<float> pre_norms;
};

static void run(
        const std::vector<struct ggml_tensor *> & v_input, // shape of v_input[0]: [n_embd, n_samples]
        const std::vector<struct ggml_tensor *> & v_output,
        layer_report * report = nullptr) {
    printf("%s: Running mean...\n", __func__);
    if (report) {
        report->n_layers = (int)v_input.size();
        report->pre_norms.resize(v_input.size());
    }
    for (size_t il = 0; il < v_input.size(); ++il) {
        // prepare output vector
        struct ggml_tensor * ctrl_out = v_output[il];
        ggml_format_name(ctrl_out, "direction.%zu", il+1);

        // calculate mean vector
        struct ggml_tensor * t_layer = v_input[il];
        GGML_ASSERT(t_layer->ne[0] == ctrl_out->ne[0]); // == n_embd
        for (int ic = 0; ic < t_layer->ne[0]; ic++) {
            float f = 0.0;
            for (int ir = 0; ir < t_layer->ne[1]; ir++) {
                f += ggml_get_f32_nd(t_layer, ic, ir, 0, 0);
            }
            f /= t_layer->ne[1];
            ggml_set_f32_1d(ctrl_out, ic, f);
        }

        // capture pre-normalization norm
        float pre_norm = 0.0;
        for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
            float f = ggml_get_f32_1d(ctrl_out, i);
            pre_norm += f*f;
        }
        pre_norm = sqrt(pre_norm);
        if (report) {
            report->pre_norms[il] = pre_norm;
        }

        // normalize output vector
        float norm = 0.0;
        for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
            float f = ggml_get_f32_1d(ctrl_out, i);
            norm += f*f;
        }
        norm = sqrt(norm);
        for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
            float f = ggml_get_f32_1d(ctrl_out, i);
            ggml_set_f32_1d(ctrl_out, i, f / norm);
        }

        printf("%s: Done layer %d / %d (pre-norm: %.4f)\n", __func__, (int) il+1, (int) v_input.size(), pre_norm);
    }
    // stampa report ordinato
    if (report && report->n_layers > 0) {
        printf("\n%s: === LAYER SIGNAL REPORT (pre-normalization norms) ===\n", __func__);
        printf("%s: Layer | Pre-norm | %% of max | Cumulative %%\n", __func__);
        printf("%s: ------+----------+----------+---------------\n", __func__);
        // calcola max per percentuali
        float max_norm = 0;
        for (auto & n : report->pre_norms) {
            if (n > max_norm) max_norm = n;
        }
        // ordina per norma decrescente
        std::vector<std::pair<int, float>> sorted;
        for (int i = 0; i < report->n_layers; i++) {
            if (report->pre_norms[i] > 0) {
                sorted.push_back({i+1, report->pre_norms[i]});
            }
        }
        std::sort(sorted.begin(), sorted.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
        float total = 0;
        for (auto & [l, n] : sorted) total += n;
        float cum = 0;
        for (auto & [layer, norm] : sorted) {
            cum += norm;
            printf("%s:   %3d   | %8.4f | %6.2f%%  | %6.2f%%\n",
                __func__, layer, norm, norm/max_norm*100, cum/total*100);
        }
        printf("%s: === END LAYER REPORT ===\n", __func__);
    }
}

}

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

model_dict = {
    "GoalGAN": "jsons/GoalGAN_metrics.json",
    "Endpoint VAE": "jsons/per_command_metrics_pecnet.json",
    "MDN": "jsons/MDN_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_3e-05_bs_16_epochs_50_height_200_width_300_mdn_type_dependent_conv_decay_5.0_num_components_3_metrics.json",
    "NonParametric": "jsons/NonParametric_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_3e-05_bs_16_epochs_50_height_200_width_300_gaussian_size_11_gaussian_sigma_3_metrics.json",
    "SinglePoint": "jsons/SinglePoint_val_monitor_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_3e-05_bs_16_epochs_50_height_200_width_300_metrics.json",
    "UnimodalNormal": "jsons/DistributionPrediction_Talk2Car_Detector_backbone_ResNet-18_use_ref_obj_True_lr_0.0001_bs_16_epochs_50_height_200_width_300_mvg_type_independent_conv_decay_1.0_metrics.json",
    "PDPC - Base": "jsons/per_command_metrics_fullconv.json",
    "PDPC - Top 64": "jsons/per_command_metrics_fullconv_topk_64.json",
    "PDPC - Top 32": "jsons/per_command_metrics_fullconv_topk_32.json",
    "RegFlow": "jsons/regression_flow_metrics_w150_h100.json",
}

intent_mapping = {
'Left': "Turn Left",
'Right': "Turn Right",
'ChangeLaneLeft': "Change Lane Left",
'ChangeLaneRight': "Change Lane Right",
 'ULeft': "U-Turn Left",
 'URight': "U-Turn Right",
 'Park': "Park",
 'Stop': "Stop",
 'PickUp': "Pick Up",
 'Continue': "Continue",
 'Overtake': "Overtake",
 'DropOff': "Drop off",
 'Follow': "Follow",
 'SlowDown': "Slow Down",
 'Wait': "Wait",
 'Approach': "Approach",
 'MoveAway': "Move Away",
 'Other': "Other",
'OtherText': None # This intent is never chosen.
}


def create_latex_table_for_intents():

    row_names = None
    model_intent_data = {}
    for model_name, model_json in model_dict.items():
        if model_json is None: continue
        counter = None

        model_data = json.load(open(model_json,"r"))
        for command_token, command_metrics in model_data.items():
            if counter is None:
                counter = {k.replace("action", ""): {"pa_2": [], "pa_4": [], "demd": [], "ade": []} for k in intents[command_token].keys()}
            if row_names is None:
                row_names = [k.replace("action", "") for k in intents[command_token].keys() if k != "actionOtherText"]

            command_intent = test_intents[command_token]
            for k,v in command_metrics.items():
                if k == "pa_2.0": k = "pa_2"
                if k == "pa_4.0": k = "pa_4"
                if k == "nll": continue
                #if model_name == "RegFlow" and "pa" in k:
                #    v /= 100
                if "pa" in k:
                    counter[command_intent][k.replace("action", "")].append(v*100)
                else:
                    counter[command_intent][k.replace("action", "")].append(v)


        # Condense information
        model_intent_data[model_name] = counter
        # for intent, metrics in counter.items():
        #     #if "Park" in intent:
        #     #print(f"[Model: {model_name}] For intent: {intent}, ADE: {np.mean(metrics['ade']) if len(metrics['ade']) > 0 else -1},"
        #     #      f"DEMD: {np.mean(metrics['demd']) if len(metrics['demd']) > 0 else -1}, PA_2: {np.mean(metrics['pa_2']) if len(metrics['pa_2']) > 0 else -1},"
        #     #      f"PA_4: {np.mean(metrics['pa_4']) if len(metrics['pa_4']) > 0 else -1}")
        #     pass


    # Output to latex
    for metric in ["ade", "demd", "pa_2", "pa_4", "made"]:
        df_dict = {}#{"intent": row_names}
        column_names = ['SinglePoint','NonParametric','UnimodalNormal','MDN',
                                                     'GoalGAN', 'Endpoint VAE',
                                                     'RegFlow', 'PDPC - Base', 'PDPC - Top 64', 'PDPC - Top 32']
            #, "\delta 2nd Best"]
        #list(model_intent_data.keys())
        for model, intent_data in model_intent_data.items():
            if metric == "demd" and model == "SinglePoint":
                column_names.pop(column_names.index(model))
                continue
            tmp = {}
            for k,v in intent_data.items():
                if metric == "made":
                    tmp[k] = np.median(v["ade"])
                else:
                    tmp[k] = np.mean(v[metric])
            df_dict[model] = tmp

        # create latex code
        columns = "|".join(["c" for _ in range(len(row_names)+1)])
        print("""
        \\begin{table*}
        \centering
        \\begin{adjustbox}{width=0.9\linewidth}
        \\begin{tabular}{|""" + columns+
        """|}\hline
        \diaghead{
        \\theadfont Diag Column}%
          {Intent}{Model}&
        """ + "&".join(["\\thead{" + name.replace("PDPC", "\\gls{destination_predictor}") + "}" for name in column_names]) + "\\\\ \hline"
              )
        for row_name in row_names:
            out = []
            for model_name in column_names:
                if model_name == "\\delta 2nd Best": continue
                out.append(df_dict[model_name][row_name])

            out_latex = [intent_mapping[row_name]]
            low_to_high_ix = np.argsort(out)
            if metric == "ade" or metric == "made":
                #ix = np.argmin(out)
                best_ix = low_to_high_ix[0]
                second_best_ix = low_to_high_ix[1]
            else:
                #ix = np.argmax(out)
                best_ix = low_to_high_ix[-1]
                second_best_ix = low_to_high_ix[-2]

            # Add the delta column value
            #out.append(out[best_ix] - out[second_best_ix])

            for out_ix, out_v in enumerate(out):
                if metric == "demd":

                    if out_ix == best_ix:
                        out_latex.append("\\textbf{" +"{:.4f}".format(out_v) + "}")
                    else:
                        out_latex.append("{:.4f}".format(out_v))

                else:

                    if out_ix == best_ix:
                        out_latex.append("\\textbf{" +"{:.2f}".format(out_v) + "}")
                    else:
                        out_latex.append("{:.2f}".format(out_v))

            print("&".join(out_latex)+ "\\\\")



        print("\hline\n"
              "\end{tabular}\n"
              "\end{adjustbox}\n"
              "\caption{" + metric +"}\n"
              "\label{}\n"
              "\end{table*}\n")

def create_box_plot(metric_name):
    model_data = []
    for model_name, model_json in model_dict.items():
        if model_json is None: continue
        #counter = None
        tmp = {"name": model_name,
               "values": []}
        data = json.load(open(model_json, "r"))
        for command_token, command_metrics in data.items():

            command_intent = test_intents[command_token]
            for k, v in command_metrics.items():
                if k == "pa_2.0": k = "pa_2"
                if k == "pa_4.0": k = "pa_4"
                if k == "nll": continue
                # if model_name == "RegFlow" and "pa" in k:
                #    v /= 100
                if k == metric_name and v < 10:

                    tmp["values"].append(v)
        model_data.append(tmp)

    df = pd.DataFrame(model_data)
    # ax = df.boxplot(grid=False, fontsize=15, sym='.',
    #                 column=['SinglePoint','NonParametric','UnimodalNormal','MDN',
    #                                                  'GoalGAN', 'Endpoint VAE',
    #                                                  'RegFlow', 'PDPC', 'PDPC - Top 64', 'PDPC - Top 32'])
    #
    # #fig, ax = plt.subplots()
    # xticklabels = ax.get_xticklabels()
    # ax.set_xticklabels(xticklabels, rotation=45,
    #                    ha='right', rotation_mode='anchor')

    # boxplot
    result = df.explode('values').reset_index(drop=True)
    result = result.assign(names=result['name'].astype('category'),
                           values=result['values'].astype(np.float32))

    ax = sns.violinplot(x='names', y='values', data=result, order=['SinglePoint','NonParametric','UnimodalNormal','MDN',
                                                     'GoalGAN', 'Endpoint VAE',
                                                     'RegFlow', 'PDPC', 'PDPC - Top 64', 'PDPC - Top 32'])

    # add stripplot
    #ax = sns.stripplot(x='names', y='values', data=result, color="orange", jitter=0.2, size=2.5)

    #for i, label in enumerate(labels)  :
    #    label.set_y(label.get_position()[1] - (i % 2) * 0.075)

    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45,
                       ha='right', rotation_mode='anchor')
    plt.savefig("violinplot_{}.png".format(metric_name),
                bbox_inches="tight", ha="right")

    #boxplot.save("boxplot_{}.png".format(metric_name))

def compute_results_for_paper():
    model_data = {}
    for model_name, model_json in model_dict.items():
        if model_json is None: continue
        # counter = None
        model_data[model_name] = {}
        data = json.load(open(model_json, "r"))
        for command_token, command_metrics in data.items():

            command_intent = test_intents[command_token]
            for k, v in command_metrics.items():
                if k == "pa_2.0": k = "pa_2"
                if k == "pa_4.0": k = "pa_4"
                if k == "nll": continue
                # if model_name == "RegFlow" and "pa" in k:
                #    v /= 100
                if k not in model_data[model_name]:
                    model_data[model_name][k] = []

                if "pa" in k:
                    v *= 100

                model_data[model_name][k].append(v)

    for model_name, data in model_data.items():
        print("*********** Metrics for [{}] ***********".format(model_name))
        for metric_name, metric_data in data.items():
            print("{}: {}, std {}: {}".format(metric_name, np.mean(metric_data),
                                              metric_name, np.std(metric_data) / np.sqrt(len(metric_data)) *2))
            #print("")

if __name__ == "__main__":
    intents = json.load(open("all_command_intents.json", "r"))
    test_intents = {k: [intent.replace("action", "") for intent, is_gt in v.items() if is_gt][0] for k, v in
                    intents.items()}

    compute_results_for_paper()

    # Create box plot
    #create_box_plot("ade")

    # Create latex table
    create_latex_table_for_intents()

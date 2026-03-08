import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load the SOTA benchmarks
combined_df = pd.read_csv('combined_results_v2.csv')
combined_df['Dataset'] = combined_df['Dataset'].str.lower()
combined_df['Dataset'] = combined_df['Dataset'].str.replace("fgvcaircraft", "aircraft")
combined_df['Dataset'] = combined_df['Dataset'].str.replace("oxfordpets", "oxfordpet")
combined_df['Dataset'] = combined_df['Dataset'].str.replace("UCF101", "ucf101")

# 2. Function to collect your experiment results across shots
def collect_my_results(dataset, shots=[1, 2, 4, 8, 16]):
    siglip_data = []
    bisiglip_data = []

    for shot in shots:
        file_name = f'c_experiment_results_{shot}_vit_b_16_siglip.csv'
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            # We take the average across the dataset for the global view 
            # or you can filter for a specific dataset like 'aircraft'
            # siglip_data.append(df['Vanilla_Acc'].mean())
            # bisiglip_data.append(df['Bilinear_Acc'].mean())
            if dataset == "Average":

                siglip_data.append(df["Vanilla_Acc"].mean())
                bisiglip_data.append(df["Bilinear_Acc"].mean())

            else:
                siglip_data.append(df.loc[df['Dataset'] == dataset, 'Vanilla_Acc'].values[0])
                bisiglip_data.append(df.loc[df['Dataset'] == dataset, 'Bilinear_Acc'].values[0])
        else:
            # Placeholder if a file is missing
            siglip_data.append(None)
            bisiglip_data.append(None)
    return siglip_data, bisiglip_data


# 3. Function to collect your experiment results across shots
def collect_my_results_clip(dataset, shots=[1, 2, 4, 8, 16]):
    clip_data = []
    biclip_data = []

    for shot in shots:
        file_name = f'c_experiment_results_{shot}_vit_b16.csv'
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            # print(df.loc[df['Dataset'] == 'imagenet', 'Vanilla_Acc'].values[0])
            # We take the average across the dataset for the global view
            # or you can filter for a specific dataset like 'aircraft'
            if dataset == "Average":
                # print(df["Vanilla_Acc"].mean(), df["Bilinear_Acc"].mean())

                clip_data.append(df["Vanilla_Acc"].mean())
                biclip_data.append(df["Bilinear_Acc"].mean())
            else:
                clip_data.append(df.loc[df['Dataset'] == dataset, 'Vanilla_Acc'].values[0])
                biclip_data.append(df.loc[df['Dataset'] == dataset, 'Bilinear_Acc'].values[0])

        else:
            # Placeholder if a file is missing
            clip_data.append(None)
            biclip_data.append(None)

    return clip_data, biclip_data


# 3. Plotting Function
def plot_few_shot_results(target_dataset='Average', plot_siglip=True):
    plt.figure(figsize=(10, 6))
    shots = [1, 2, 4, 8, 16]
    x_labels = ['1-shot', '2-shot', '4-shot', '8-shot', '16-shot']

    # Plot SOTA Models from combined_results_v2
    # Filtering for the dataset (or average of the dataset)
    sota_methods = ['Linear probe CLIP', 'CoOp', 'CoCoOp', 'MaPLe', 'PromptSRC']
    # sota_methods = ['CoOp', 'CoCoOp', 'MaPLe']
    for method in sota_methods:

        data = combined_df[(combined_df['Method'] == method)]

        if target_dataset != 'Average':
            data = data[data['Dataset'] == target_dataset]
        try:
            y_values = data.iloc[0, 2:7].values  # Columns for 1, 2, 4, 8, 16 shots
        except:
            pass
        plt.plot(x_labels, y_values, marker='o', label=method, linestyle='--')

    # Add your BiSigLIP and SigLIP Baseline
    # (Using the 1-shot file you uploaded as a template for the shot logic)

    clip_y, biclip_y = collect_my_results_clip(target_dataset, shots)
    plt.plot(-0.1, clip_y[0], marker='*', markersize=10, label='CLIP (Baseline)', linewidth=2, linestyle='--')
    plt.plot(x_labels, biclip_y, marker='s', markersize=10, label='BiCLIP (Ours)', linewidth=3, color='black')

    if plot_siglip:
        sig_y, bi_y = collect_my_results(target_dataset, shots)

        plt.plot(-0.1, sig_y[0], marker='*', markersize=10, label='SigLIP (Baseline)', linewidth=2, linestyle='--')
        plt.plot(x_labels, bi_y, marker='s', markersize=10, label='BiSigLIP (Ours)', linewidth=3, color='red')

    plt.title(f'Few-Shot Performance Comparison: {target_dataset}', fontsize=20)
    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy (%)')
    plt.legend(fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.show()
    plt.savefig(f"plots/{target_dataset}_few_shot_comparison.png")


# # Run for a specific dataset or the global average
# plot_few_shot_results('Average')
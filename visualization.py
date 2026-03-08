import argparse
import torch
from visualization.angular_distribution import angular_distribution
from visualization.fewshot import plot_few_shot_results
from PIL import ImageFile

from visualization.orthognality import orthogonality_analysis

device = "cuda" if torch.cuda.is_available() else "cpu"

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-f', '--few-shot', action='store_true',
                        help='Visualize Few-Shot tasks.')
    parser.add_argument('-a', '--angular-dist', action='store_true',
                        help='Plot angular distribution.')
    parser.add_argument('-o', '--orthogonality', action='store_true',
                        help='Compute orthogonality.')


    args = parser.parse_args()

    datasets = ["Average",
                "aircraft",
                "oxfordpet",
                "flowers102",
                "stanfordcars",
                "food101",
                "dtd",
                "eurosat",
                "sun397",
                "caltech101",
                "ucf101",
                "imagenet"
                ]

    if args.few_shot:

        for ind, k in enumerate(datasets):


            try:
                plot_few_shot_results(k, plot_siglip=True)

                print(f"{ind+1} --> {k}: Visualizing Few-Shot tasks")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                print(f"{ind+1} --> {k}: Failed to visualize Few-Shot tasks")
                # exit()
    if args.angular_dist:

        for ind, k in enumerate(datasets[5:]):
            if k == "Average":
                continue

            print(f"Plotting angular distribution for {k}")
            angular_distribution(k)
            print(f"----------- Done ! ---------------")

    if args.orthogonality:
        for k in datasets:
            if k == "Average":
                continue

            print(f"Plotting orthogonality for {k}")
            orthogonality_analysis(k, device)

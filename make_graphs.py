import sys
import re
import os
import argparse


def main(filename):
    if not os.path.exists(filename):
        print("File not found")
        return
    with open(filename, "r") as f:
        lines = f.readlines()
    left_parts, right_parts = {}, {}
    left_counter, right_counter = 1, 1
    for line in lines:
        if "label=" in line:
            if not "/" in line:
                continue
            try:
                label = re.search(r'label="(.+?)"', line).group(1)
                left_part = label.split("/")[0]
                right_part = label.split("/")[1]
            except IndexError:
                print("IndexError", label, "in", filename)
                exit()
            if left_part not in left_parts:
                left_parts[left_part] = "I" + str(left_counter)
                left_counter += 1
            if right_part not in right_parts:
                right_parts[right_part] = "O" + str(right_counter)
                right_counter += 1
    # sort the parts from longer to shorter, because we want to replace the longer parts first
    left_parts = dict(
        sorted(left_parts.items(), key=lambda item: len(item[0]), reverse=True)
    )
    right_parts = dict(
        sorted(right_parts.items(), key=lambda item: len(item[0]), reverse=True)
    )
    for left_part in left_parts:
        for i, line in enumerate(lines):
            lines[i] = line.replace('"' + left_part, '"' + left_parts[left_part])
    for right_part in right_parts:
        for i, line in enumerate(lines):
            lines[i] = line.replace(right_part + '"', right_parts[right_part] + '"')
    with open(filename.replace(".dot", "_simplified.dot"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify the labels of a dot file")
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        help="The directory where the dot files are stored",
    )
    parser.add_argument(
        "-b",
        "--base_method",
        type=str,
        choices=["state_coverage", "wmethod", "wpmethod", "rwpmethod"],
        help="The base method name",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=str,
        choices=["tls", "mqtt", "tcp", "dtls"],
        help="The protocol",
    )
    parser.add_argument(
        "-t",
        "--trial",
        type=str,
        choices=list(map(lambda x: str(x), range(30))),
        help="The trial of the experiment. 0 for deterministic method"
    )
    args = parser.parse_args()

    directory = args.results_dir + '/' + args.base_method + '/' + args.protocol.upper()

    for item in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, item)):
            continue
        trial = os.path.join(directory, item, "trial_" + args.trial)
        oracles = os.listdir(trial)
        for oracle in oracles:
            hypotheses = os.listdir(os.path.join(trial, oracle))
            for files in filter(lambda x: x.endswith('.dot'), hypotheses):
                main(os.path.join(trial, oracle, files))

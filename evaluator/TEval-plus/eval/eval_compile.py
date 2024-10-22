import logging
from joblib import Parallel, delayed
import glob, os, argparse
import subprocess
from tqdm import tqdm
import pandas
import csv

def run_complie(project, gen_dir, result_dir, timeout, args, i, tot):
    cmd = f'run_coverage.pl -p {project} -d {gen_dir} -o {result_dir} -t /home/tmp'
    success = True
    is_time_out = False
    error_msg = None
    num_fail=0
    if args.v:
        logging.info(f'running ({i}/{tot}):', cmd)
    try:
        res = subprocess.run(cmd.split(), capture_output=True, timeout=timeout)
        stdout = res.stdout.decode('utf-8')
        stderr = res.stderr.decode('utf-8')
#        num_ok = stderr.count('.OK')
        num_fail = stderr.count('FAIL')
        if 'FAILED' in stdout or 'FAILED' in stderr:
            success = False
            error_msg = stderr
    except subprocess.TimeoutExpired:
        is_time_out = True

    return gen_dir, success, is_time_out, error_msg,num_fail

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_test_dir')
    parser.add_argument('-o', dest='result_dir', default='results')
    parser.add_argument('-t', '--timeout', dest='timeout', type=int, default=180)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-n', '--njobs', dest='njobs', type=int, default=None)
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()

    gen_test_dir = args.gen_test_dir
    result_dir = args.result_dir
    njobs = args.njobs

    if os.path.exists(result_dir):
        subprocess.run(f'rm -r {result_dir}'.split())

    projects_dirs = glob.glob(gen_test_dir + '/*')

    task_args = []
    for pd in projects_dirs:
        p = os.path.basename(pd)

        for gen_dir in glob.glob(f'{pd}/*/*'):
            if glob.glob(gen_dir + '/*.tar.bz2'):
                task_args += [(p, gen_dir, result_dir)]

    tot = len(task_args)
    if args.test:
        task_args = task_args[:32]
    tasks = [delayed(run_complie)(*task_arg, args.timeout, args, i + 1, tot) for i, task_arg in enumerate(task_args)]
    if not args.v:
        results = Parallel(n_jobs=njobs, prefer='processes')(tqdm(tasks))
    else:
        results = Parallel(n_jobs=None, prefer='processes')(tqdm(tasks))


    err_log = []
    item = []
    all_num, num_pass, all_num_fail, all_line_ratio, all_branch_ratio, syn_err, comp_err = 0, 0, 0, 0, 0, 0, 0
    for result in results:
        all_num+=1
        cur_gen_dir, success, is_time_out, error_msg,num_fail = result
        if success and not is_time_out:
            continue
        if num_fail > 0:
            err_log.append(error_msg)
            all_num_fail += 1
    coverage_dir = f'{result_dir}coverage'
    with open(coverage_dir) as file:
        contents = file.readlines()
    for i in range(1, len(contents)):
        ratio_line = 0.0
        ratio_branch = 0.0
        metrics = contents[i].strip().split(",")  # 移除行末尾的换行符并分割
        if len(metrics) == 1:
            syn_err += 1
        if len(metrics) > 1 and ('-' in contents[i]):
            comp_err += 1
        if len(metrics) > 1 and ('-' not in contents[i]):
            num_pass += 1

    # 创建另一个 CSV 文件并写入错误统计数据
    with open(f'{result_dir}/compile_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['tot','all_num','all_num_fail','syn_err', 'comp_err', 'num_pass']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'tot':tot,
            'all_num':all_num,
            'all_num_fail':all_num_fail,
            'syn_err': syn_err,
            'comp_err': comp_err,
            'num_pass': num_pass,
        })

    with open(os.path.join(result_dir, "failed_tests_log.csv"), 'w', encoding='utf-8') as f:
        for item in err_log:
            item = str(item).replace("\n", "")
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()



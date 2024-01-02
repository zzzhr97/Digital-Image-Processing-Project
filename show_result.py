from utils import restore_results, visualize_results, save_results_with_writer

# if True, load from ./results/result_task/result_net/result_name
# otherwise, load from ./
is_from_dir = False

# the save path of tensorboard file 
save_path = './log/tmp'

result_name = 'lr1e-06_bs4_epochs200_seed512'   # the name of .csv file
result_task = 1
result_net = 'ResNet50'

result_path = f'./results/task-{result_task}/{result_net}/{result_name}.csv' if is_from_dir else f'./{result_name}.csv'

def show_result():
    result = restore_results(result_path)
    visualize_results(result, None)
    save_results_with_writer(result, save_path)

def main():
    show_result()

if __name__ == '__main__':
    main()
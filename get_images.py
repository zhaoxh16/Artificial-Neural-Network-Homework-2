import os

first_grades = ["cnn", "cnn_no_bn", "mlp", "mlp_no_bn"]
second_grades = ["model_1", "model_2", "model_3", "model_4", "model_5"]

for first_grade in first_grades:
    for second_grade in second_grades:
        acc_dir = os.path.join(os.path.join(first_grade, "result"), os.path.join(second_grade, "acc.png"))
        loss_dir = os.path.join(os.path.join(first_grade, "result"), os.path.join(second_grade, "loss.png"))
        target_acc_dir = os.path.join("images", first_grade+'_'+second_grade+'_'+'acc.png')
        target_loss_dir = os.path.join("images", first_grade+ '_' + second_grade + '_' + 'loss.png')
        print(acc_dir, loss_dir)
        os.system('cp ' + acc_dir + ' ' + target_acc_dir)
        os.system('cp ' + loss_dir + ' ' + target_loss_dir)

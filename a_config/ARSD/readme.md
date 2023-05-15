1、dota_model_area
student 
2_data b=8 [20,28]-30 lr=0.005

teacher 
1_data b=8 [20,28]-30 lr=0.005 数据增强
2_data b=8 [20,28]-30 lr=0.005 
3_data b=4 [16,22]-24 lr=0.001

dis 
1_data b=8 [20,28]-30 lr=0.005 18_0.5_area
2_data b=8 [20,28]-30 lr=0.005 18_0.5
3_data b=8 [20,28]-30 lr=0.005 18

dis_no_gn




2、dota_new_model
student 1_data b=8 [20,28]-30 lr=0.005 数据增强

teacher 1_data b=8 [20,28]-30 lr=0.005 数据增强

dis     1_data b=8 [20,28]-30 lr=0.005 数据增强-对蒸馏影响严重。
dis     2_data b=8 [16,22]-24 lr=0.001 


在自己采用大图测试的时候，把测试集的里面的东西替换为验证集。
modes = ["gen", "dis"]

batch_size_gen = 64  
batch_size_dis = 64  
lr_gen = 1e-3 
lr_dis = 1e-3
n_epochs = 30 
n_epochs_gen = 5  
n_epochs_dis = 5 


load_model = False  
save_steps = 1  


n_emb = 50


# diffusion path
train_cascades = "data/example_train_cascades"
test_cascades = "data/example_test_cascades"
small_train_cascades = "data/small_train_cascades"
small_test_cascades = "data/small_test_cascades"
model_log = "log/"

emb_filenames = ["results/diff_gen_.emb",
                 "results/diff_dis_.emb"]
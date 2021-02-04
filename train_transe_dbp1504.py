import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

def main():
	print("Started!")
	print("Configuring train data loader")
	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = "./dbpedia-2015-04/",
		nbatches = 10,
		threads = 8,
		sampling_mode = "normal",
		bern_flag = 1,
		filter_flag = 1,
		neg_ent = 25,
		neg_rel = 0)

	# dataloader for test
	#test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")
	print("Configuring TransE model")
	# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 100,
		p_norm = 1,
		norm_flag = True)

	print("Configuring loss function")
	# define the loss function
	model = NegativeSampling(
		model = transe,
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)
	print("Starting training")
	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 5, alpha = 1.0, use_gpu = True)
	trainer.run()
	print("Training finished. Saving model.")
	transe.save_checkpoint('./checkpoint/transe_dbpedia201504.ckpt')
	transe.save_parameters('./dbpedia201504_transe.json')
	print("Done!")
# test the model
#transe.load_checkpoint('./checkpoint/transe.ckpt')
#tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
#tester.run_link_prediction(type_constrain = False)
if __name__ == "__main__":
    main()
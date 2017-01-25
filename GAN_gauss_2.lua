
  require 'nn'
  require 'gnuplot'
 require 'nngraph'
 require 'distributions'

 ---------------------------------------------------------
 ------------- COMMAND OPTIONS ---------------------------
 ---------------------------------------------------------

 cmd = torch.CmdLine()
 ------------- Algorithm ------------
 cmd:option('-batch_size',25,"mini-batch size")
 cmd:option('-maxEpoch',1000,"number of epochs")
 cmd:option('-lr',1e-3,"learning rate")
 cmd:option('-step', 1, "number of discriminator training iteration for one generative training iteration")

 ------------- Data -----------------
 cmd:option('-DIMENSION', 1, "dimension of the example data")
 cmd:option('-n_points', 1000, "number of examples")
 cmd:option('-ratio',0.8,"train/total ratio. To split the dataset in train and test sets")
 cmd:option('-mean', 1, "mean of the Gaussian distribution to sample from")
 cmd:option('-var', 1, "variance of the Gaussian distribution to sample from")
 ------------ Model -----------------
 cmd:option('-noise_size', 1, "dimension of the noise vector")
 cmd:option('-noise_type', "Gaussian", "either Gaussian or Uniform")
 cmd:option('-noise_mean',4 , "mean value for the noise distribution")
 cmd:option('-noise_var', 0.5, "variance for the noise distribution")
 cmd:option('-generative_size', 10, "dimension of the hidden layers of the generative model")
 cmd:option('-discrim_size',10, "dimension of the hidden layers of the discriminant model")

 local opt = cmd:parse(arg)
 print("GAN Implementation with Gaussian distributed data")
 print("Parameters of this experiment :")
 print(opt)

batch_dep=opt.n_points/opt.batch_size -- Warning : Size of batch !
-- Tirage d'une gaussienne
local x_train_d=torch.Tensor(opt.n_points,opt.DIMENSION)
for i=1,opt.n_points do
  x_train_d[i]:copy(torch.randn(opt.DIMENSION)*opt.var+opt.mean);
end

z_noise=torch.Tensor(batch_dep,opt.DIMENSION)
label_dat=torch.Tensor(batch_dep,opt.DIMENSION)
label_dat_false=torch.Tensor(batch_dep,opt.DIMENSION)
z_noise_final=torch.Tensor(opt.n_points,opt.DIMENSION)

-- Creating noise
function create_noise()
  for i=1,batch_dep/2 do
    z_noise[i]=torch.uniform(-1,1)
  end
     return z_noise
end

--Creating labels real data for discriminative model
for i=1,batch_dep/2 do
  label_dat[i]=1
end
--Creating labels fake data for discriminative model
for i=batch_dep/2+1,batch_dep do
  label_dat[i]=0
end
--Creating labels fake data generative model
for i=1,batch_dep do
  label_dat_false[i]=1
end
-- Creating noise for eval
  for i=1,opt.n_points do
     z_noise_final[i]=torch.uniform(-1,1)
  end


--------------------------------------------
--------------CREATE MODEL G----------------
--------------------------------------------
local model_g=nn.Sequential()
model_g:add(nn.Linear(opt.DIMENSION,opt.generative_size))
model_g:add(nn.SoftPlus())
model_g:add(nn.Linear(opt.generative_size,1))


--------------------------------------------
--------------CREATE MODEL D----------------------
--------------------------------------------
local model_d=nn.Sequential()
model_d:add(nn.Linear(opt.DIMENSION,opt.discrim_size))
model_d:add(nn.ReLU())
model_d:add(nn.Linear(opt.discrim_size,opt.discrim_size))
model_d:add(nn.ReLU())
model_d:add(nn.Linear(opt.discrim_size,1))
model_d:add(nn.Sigmoid())



function Prepare_batch()
      shuffle = torch.randperm(x_train_d:size(1))
      x_batch_d = shuffle:chunk(opt.batch_size,1)
      return x_batch_d
end

--------------------------------------------
--------------TESTING ----------------------
--------------------------------------------

function Eval_model(iteration)
  if(iteration%100==0) then
    output=model_g:forward(z_noise_final)
    gnuplot.hist(output)
  end
end


---------------------------------------------
-------------- TRAINNING  -------------------
---------------------------------------------
local criterion=nn.BCECriterion()

local all_losses={}
local all_losses_g={}

for iteration=1,opt.maxEpoch do
    for k=1,opt.step do
      x_batch_d=Prepare_batch()
      loss_d=0
      loss_g=0
      Eval_model(iteration)
      --gnuplot.hist(model_g:forward(z_noise_final))
      for i=1,opt.batch_size do
        model_d:zeroGradParameters()
        -- Forward noise
        z=create_noise()
        data_gen=model_g:forward(z)

        new_batch=x_train_d:index(1,x_batch_d[i]:long())
        -- Concatenate real and fake data
        new_batch:sub(batch_dep/2+1,batch_dep):copy(data_gen:sub(1,batch_dep/2))
        -- compute the discriminator decision
        output=model_d:forward(new_batch)
        --  loss of discriminative decision
        loss_d=criterion:forward(output,label_dat)
        -- Pass backward
        delta=criterion:backward(output,label_dat)
        model_d:backward(new_batch,delta)
        -- update parameters generator
        model_d:updateParameters(opt.lr)

      end
      table.insert(all_losses,loss_d)
      --gnuplot.plot('train_d loss',torch.Tensor(all_losses))

    end


    x_batch=Prepare_batch()

    for i=1,opt.batch_size do
      -- Remise à zeros des parametres
      model_g:zeroGradParameters()
      model_d:zeroGradParameters()
      -- Forward noise
      z=create_noise()
      data_gen=model_g:forward(z)
      -- Forward data_gen in model D
      output=model_d:forward(data_gen)
      -- Forward avec le criterion BCE
      loss_g=criterion:forward(output,label_dat_false)
      delta=criterion:backward(output,label_dat_false)
      -- update for back_prop data_gen
      delta_d=model_d:backward(data_gen,delta)
      model_d:zeroGradParameters()
      -- update for back_prop generator
      model_g:backward(z,delta_d)
      -- update parameters generator
      model_g:updateParameters(opt.lr)
    end
    -- insert generator loss in table
    table.insert(all_losses_g,loss_g)
    -- display generator histogram
    --gnuplot.hist(model_g:forward(z_noise_final))

end
--gnuplot.hist(model_g:forward(z_noise_final))
--gnuplot.plot('train loss',torch.Tensor(all_losses))
gnuplot.plot('train loss',torch.Tensor(all_losses_g))
--gnuplot.plot('test',)

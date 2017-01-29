
  require 'nn'
 require 'optim'
 require 'gnuplot'
 require 'utils'
require 'distributions'
require 'randomkit'
require 'itorch'
 ---------------------------------------------------------
 ------------- COMMAND OPTIONS ---------------------------
 ---------------------------------------------------------

 cmd = torch.CmdLine()
 ------------- Algorithm ------------
 cmd:option('-batch_size',100,"mini-batch size")
 cmd:option('-maxEpoch',5000,"number of epochs")
 cmd:option('-lr',1e-4,"learning rate")
 cmd:option('-step', 1, "number of discriminator training iteration for one generative training iteration")

 ------------- Data -----------------
 cmd:option('-DIMENSION', 2, "dimension of the example data")
 cmd:option('-n_points',1000, "number of examples")
 cmd:option('-ratio',0.8,"train/total ratio. To split the dataset in train and test sets")
 cmd:option('-mean', 8, "mean of the Gaussian distribution to sample from")
 cmd:option('-var', 0.5, "variance of the Gaussian distribution to sample from")
 ------------ Model -----------------
 cmd:option('-noise_size', 10, "dimension of the noise vector")
 cmd:option('-noise_type', "Gaussian", "either Gaussian or Uniform")
 cmd:option('-generative_size',40, "dimension of the hidden layers of the generative model")
 cmd:option('-discrim_size',20, "dimension of the hidden layers of the discriminant model")
 cmd:option('-var_noise', 0.5, "mean of the noise distribution")
 cmd:option('-mean_noise', 4, "mean of the noise distribution")
 cmd:option('-seed_value', 1010, "seed value for random generated data")


 local opt = cmd:parse(arg)
 print("GAN Implementation with Gaussian distributed data")
 print("Parameters of this experiment :")
 print(opt)

 -- key to comparate model with same data
 --torch.manualSeed(opt.seed_value)
 batch_dep=opt.n_points/opt.batch_size -- Warning : Size of batch !
-- Tirage d'une gaussienne
local x_train_d=torch.Tensor(opt.n_points,opt.DIMENSION)
for i=1,opt.n_points do
  x_train_d[i]:copy(torch.randn(opt.DIMENSION)*opt.var+opt.mean);
end

z_noise=torch.Tensor(batch_dep,opt.noise_size)
label_dat=torch.Tensor(2*batch_dep,opt.DIMENSION)
label_dat_test=torch.Tensor(2*x_train_d:size(1),opt.DIMENSION)
label_dat_false_test=torch.Tensor(x_train_d:size(1),opt.DIMENSION)
label_dat_false=torch.Tensor(batch_dep,opt.DIMENSION)
z_noise_final=torch.Tensor(opt.n_points,opt.noise_size)

cpt=0
--------------------------------------------------------------------------------
--------------------CREATE NOISE, REAL & FAKE LABEL ----------------------------
--------------------------------------------------------------------------------


-- Creating noise
function create_noise()
  for i=1,batch_dep do
    z_noise[i]:copy(torch.randn(opt.noise_size)*opt.var_noise+opt.mean_noise)
  end
     return z_noise
end

--Creating labels real data for discriminative model
for i=1,batch_dep do
  label_dat[i]=1
end
--Creating labels fake data for discriminative model
for i=batch_dep+1,2*batch_dep do
  label_dat[i]=0
end
--Creating labels fake data generative model
for i=1,batch_dep do
  label_dat_false[i]=1
end

--Creating labels real data for discriminative model for test
for i=1,x_train_d:size(1) do
  label_dat_test[i]=1
end
--Creating labels fake data for discriminative model for test
for i=x_train_d:size(1)+1,2*x_train_d:size(1) do
  label_dat_test[i]=0
end
--Creating labels fake data generative model for test
for i=1,x_train_d:size(1) do
  label_dat_false_test[i]=1
end

-- Creating noise for eval
function Create_noise_final()
  for i=1,opt.n_points do
     z_noise_final[i]:copy(torch.randn(opt.noise_size)*opt.var_noise+opt.mean_noise)
  end
  return z_noise_final
end

--------------------------------------------------------------------------------
-------------------------------CREATE MODEL G-----------------------------------
--------------------------------------------------------------------------------
local model_g=nn.Sequential()
model_g:add(nn.Linear(opt.noise_size,opt.generative_size))
model_g:add(nn.ReLU())
model_g:add(nn.Linear(opt.generative_size,opt.DIMENSION))


--------------------------------------------------------------------------------
-------------------------------CREATE MODEL D-----------------------------------
--------------------------------------------------------------------------------
local model_d=nn.Sequential()
model_d:add(nn.Linear(opt.DIMENSION,opt.discrim_size))
model_d:add(nn.ReLU())
model_d:add(nn.Linear(opt.discrim_size,opt.discrim_size))
model_d:add(nn.ReLU())
model_d:add(nn.Linear(opt.discrim_size,opt.DIMENSION))
model_d:add(nn.Sigmoid())

--------------------------------------------------------------------------------
--------------------------------PREPARE BATCH ---------------------------------
--------------------------------------------------------------------------------


function Prepare_batch()
      local shuffle = torch.randperm(x_train_d:size(1))
      local x_batch_d = shuffle:chunk(opt.batch_size,1)
      return x_batch_d
end

function Prepare_real_data_test()
      local x_test=torch.Tensor(opt.n_points,opt.DIMENSION)
      local shuffle = torch.randperm(x_train_d:size(1))
      local x_test_chunk = shuffle:chunk(x_train_d:size(1),1)
      for index = 1,x_train_d:size(1) do
          x_test[index] =x_train_d[shuffle[index]]
      end
      return x_test
end



--------------------------------------------------------------------------------
--------------------------------EVAL & TESTING ---------------------------------
--------------------------------------------------------------------------------
function Eval_model(i)
  local data_gen_test=model_g:forward(Create_noise_final())
  -- Concatenate the true and fake data
  data_mixte=torch.cat(x_test,data_gen_test,1)
  -- Test mixte data
  local output=model_d:forward(data_mixte)
  local loss_d_test=criterion:forward(output,label_dat_test)
  table.insert(all_losses_d_test,loss_d_test)
  --gnuplot.plot(torch.Tensor(all_losses_d_test))

  -- test data generate
  local output_fake=model_d:forward(output:sub(x_test:size(1)+1,2*x_test:size(1)))
  local loss_g_test=criterion:forward(output_fake,label_dat_false_test)
  table.insert(all_losses_g_test,loss_g_test)
  --gnuplot.plot(torch.Tensor(all_losses_g_test))


 --- Create decision boundary & display generator histogram

  if(i%100==0 and cpt==0) then
    if(opt.DIMENSION==1) then
      display_decision=torch.Tensor(200,2)
      gnuplot.axis({-2,13,0,0.12})
      for i=1,200 do
        display_decision[i][1]=0+(13-0)/200*(i-1)
        display_decision[i][2]=model_d:forward(torch.Tensor(1):fill(display_decision[i][1]))/10
      end
      gnuplot.plot({histogram(x_train_d,30),'-'},{histogram(data_gen_test,30),'-'},{display_decision,'-'})
      --cpt=1
    elseif(opt.DIMENSION==2) then
      gnuplot.axis({-1,10,-2,10})
      gnuplot.plot({'real data',x_train_d,'+'},{'generate data',data_gen_test,'+'})
      gnuplot.movelegend('left','top')



  end
end
end


--------------------------------------------------------------------------------
--------------------------- TRAINNING ------------------------------------------
--------------------------------------------------------------------------------
criterion=nn.BCECriterion()

all_losses={}
all_losses_g={}
all_losses_d_test={}
all_losses_g_test={}
decision_fake={}
decision={}
x_test=Prepare_real_data_test()
for iteration=1,opt.maxEpoch do
  if(iteration%1000==0) then
    opt.lr=opt.lr/2
  end
    for k=1,opt.step do
      x_batch_d=Prepare_batch()
      loss_d=0
      loss_g=0
      Eval_model(iteration)
      for i=1,opt.batch_size do
        model_d:zeroGradParameters()
        -- Forward noise
        z=create_noise()
        data_gen=model_g:forward(z)

        new_batch=x_train_d:index(1,x_batch_d[i]:long())
        -- Concatenate real and fake data
        new_batch2=torch.cat(new_batch,data_gen,1)
        -- compute the discriminator decision
        output=model_d:forward(new_batch2)
        --  loss of discriminative decision
        loss_d=criterion:forward(output,label_dat)
        -- Pass backward
        delta=criterion:backward(output,label_dat)
        model_d:backward(new_batch2,delta)
        -- update parameters generator
        model_d:updateParameters(opt.lr)

      end
      table.insert(all_losses,loss_d)
      --gnuplot.figure()
      --gnuplot.plot(torch.Tensor(all_losses))
      if iteration%100 == 0 then
       print("Achievement : " .. iteration/opt.maxEpoch*100 .. "%")
    end

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
      -- Backward avec le criterion BCE
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
    --gnuplot.plot(torch.Tensor(all_losses_g))

end

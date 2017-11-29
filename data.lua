require 'math'
require 'torch'
require 'math'
require 'xlua'


local D={}
function D.dataset(d_number)
	local dataNumber = d_number
	local inputSize = {10,9}
	local trainDataSet = torch.Tensor(d_number,inputSize[1],inputSize[2])
	local trainSetGT = torch.Tensor(dataNumber, inputSize[2]-1)

	for index = 1, dataNumber do 
		xlua.progress(index, dataNumber)
		local trainData = torch.Tensor(inputSize[1],inputSize[2])
		for i=1,inputSize[1] do
			for j=1,inputSize[2] do
				trainData[i][j] = math.random(10)
			end
		end
		-- print(trainData)
		u,s,v = torch.svd(trainData)
		-- print(u)
		-- print(s)
		-- print(v)
		s = torch.diag(s)
		print(s)
		s[9][9] = 0
		x = v:select(2,9)
		x_val = v:select(2,9)
		-- print(x)
		-- print(x_val)
		x = x:div(x[9])
		x_val = x_val:div(x_val[9])
		print(x)
		-- print(x_val)
		for tg_i = 1,inputSize[2]-1 do
			trainSetGT[index][tg_i] = x[tg_i]
		end
		-- print(trainSetGT[index])
		v = v:t()
		-- print(v)
		local us = torch.Tensor(inputSize[1],inputSize[2])
		for ii=1,u:size()[1] do
			for jj=1,s:size()[1] do
				z = torch.cmul(u:select(1,ii),s:select(2,jj))
				us[ii][jj] = torch.cumsum(z)[9]
			end
		end
		-- print(us)
		local usv = torch.Tensor(inputSize[1],inputSize[2])

		for iii=1,us:size()[1] do
			for jjj=1,v:size()[1] do
				zz = torch.cmul(us:select(1,iii),v:select(2,jjj))
				usv[iii][jjj] = torch.cumsum(zz)[9]
			end
		end
		-- print(usv)

		--test

		-- for i_val=1, usv:size()[1] do 
		-- 	z_val = torch.cmul(usv:select(1,i_val),x_val)
		-- 	print(torch.cumsum(z_val)[9])
		-- end

		trainDataSet[index]:copy(usv)
	end
	return trainDataSet,trainSetGT
end

local trainingData,trainingGT,testingData,testingGT
if not paths.filep(paths.concat(opt.cache, 'trainData.t7')) then
	trainingData,trainingGT = D.dataset(1)
	-- torch.save(paths.concat(opt.cache, 'trainData.t7'), trainingData)
	-- torch.save(paths.concat(opt.cache, 'trainGT.t7'), trainingGT)
else
	trainingData = torch.load(paths.concat(opt.cache, 'trainData.t7'))
	trainingGT = torch.load(paths.concat(opt.cache, 'trainGT.t7'))
end
if not paths.filep(paths.concat(opt.cache, 'testData.t7')) then
	testingData,testingGT = D.dataset(1)
	-- torch.save(paths.concat(opt.cache, 'testData.t7'), testingData)
	-- torch.save(paths.concat(opt.cache, 'testGT.t7'), testingGT)
else
	testingData = torch.load(paths.concat(opt.cache, 'testData.t7'))
	testingGT = torch.load(paths.concat(opt.cache, 'testGT.t7'))
end
return trainingData,trainingGT, testingData, testingGT



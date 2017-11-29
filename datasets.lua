require 'paths'
require 'torch'

local Data = {}
path = '/Users/bragi/Computer_Vision/MLP/data'
function Data.load(path)
	s = 0
	ss = 0
	if paths.filep(paths.concat(path, 'trainData.t7')) then
		trainingData = torch.load(paths.concat(path, 'trainData.t7'))
		trainingGT = torch.load(paths.concat(path, 'trainGT.t7'))
		testingData = torch.load(paths.concat(path, 'testData.t7'))
		testingGT = torch.load(paths.concat(path, 'testGT.t7'))
		print(trainingData:size())
		print(trainingGT:size())
		print(testingData:size())
		print(testingGT:size())
		for i = 1,trainingData:size()[1] do
			for j =1,trainingData:size()[2] do
				t = torch.cumsum(torch.cmul(trainingData[i][j],trainingGT[i]))[9]
				s = s + t
			end
		end
		print(s)
		for ii = 1,testingData:size()[1] do
			for jj =1,testingData:size()[2] do
				tt = torch.cumsum(torch.cmul(testingData[ii][jj],testingGT[ii]))[9]
				ss = ss + t
			end
		end
		print(ss)
		-- print(trainingData[1])
		-- print(trainingData[2])
		-- print(trainingGT[1])
		-- print(trainingGT[2])
		-- print(testingData[1])
		-- print(testingData[2])
		-- print(testingGT[1])
		-- print(testingGT[2])
		local trainGT = torch.Tensor(trainingGT:size()[1],trainingGT:size()[2]-1)
		local testGT = torch.Tensor(testingGT:size()[1],testingGT:size()[2]-1)

		for m = 1,trainingGT:size()[1] do
			for n=1,trainingGT:size()[2]-1 do
				trainGT[m][n] = trainingGT[m][n]
			end
		end

		for mm = 1,testingGT:size()[1] do
			for nn=1,testingGT:size()[2]-1 do
				testGT[mm][nn] = testingGT[mm][nn]
			end
		end
		print(testGT[1])
		-- torch.save('trainGT2.t7',trainGT)
		torch.save('testGT2.t7',testGT)
		-- print(trainGT[1])

	else
		print('no data found in the directoy: ' .. path)
	end
end

return Data.load(path)
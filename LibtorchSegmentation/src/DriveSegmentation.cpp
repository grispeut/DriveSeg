/******************************************
*\author runrunrun1994
*\date   2020-11-15 
*******************************************/

#include "DriveSegmentation.h"
#include "log.h"
#include "utils.h"

#include <torch/script.h>

bool DriveSegmentation::InitModel(PTModelParam pparams)
{
    //初始化模型参数
    mtParam.modelPath  = pparams->modelPath;
    mtParam.meanValue  = pparams->meanValue;
    mtParam.stdValue   = pparams->stdValue;
    mtParam.deviceType = pparams->deviceType;
    mtParam.gpuId      = pparams->gpuId;

    //判断模型文件是否存在
    bool isExistsFile = IsExists(mtParam.modelPath);

    if (!isExistsFile){
        LOG_ERROR("The file %s is not found!", mtParam.modelPath.c_str());

        return false;
    }

    std::cout << mtParam.modelPath<< std::endl;

    try{
        mpDriveSegModel = std::make_shared<torch::jit::script::Module>(torch::jit::load((mtParam.modelPath).c_str()));
    }
    catch(const c10::Error& e)
    {
        LOG_ERROR("Counld't loaded the model: %s", mtParam.modelPath.c_str());

        return false;
    }

    if (HaveGPUs()){
        mpDriveSegModel->to(torch::Device(mtParam.deviceType, mtParam.gpuId));
    }

#ifdef DEBUG
    LOG_INFO("The model is loaded!");
#endif

    mpDriveSegModel->eval();

    return true;
}

void DriveSegmentation::PreProcess(const cv::Mat& src,torch::Tensor& input)
{
    if (src.empty())
    {
        LOG_ERROR("The image is NULL!");

        return ;
    }

    cv::Mat rgbMat;
    cv::cvtColor(src, rgbMat, cv::COLOR_BGR2RGB);

    input = torch::from_blob(rgbMat.data, {1, rgbMat.rows, rgbMat.cols, 3}, torch::kByte);
    input = input.permute({0, 3, 1, 2});
    Normalize(input, mtParam.meanValue, mtParam.stdValue);
}

torch::Tensor DriveSegmentation::Forward(cv::Mat& img)
{
    torch::Tensor input;
    PreProcess(img, input);
    if (HaveGPUs())
        input = input.to(torch::Device(mtParam.deviceType, mtParam.gpuId));

    torch::NoGradGuard no_grad;
    //auto output = mpDriveSegModel->forward({input}).toTensor().squeeze(0).argmax(0).cpu().clamp(0, 255).to(torch::kU8);
    auto output = mpDriveSegModel->forward({input.to(torch::kHalf)}).toTensor().squeeze(0).argmax(0).cpu().clamp(0, 255).mul(255.).to(torch::kU8);
    //std::cout<<output;

    return output;
}

bool DriveSegmentation::UnInitModel()
{
    mpDriveSegModel.reset();
}

DriveSegmentation::~DriveSegmentation()
{
    UnInitModel();
}

console.log("handwrite.js 已成功加载");

let mnistModel = null;
let mnistReady = false;

// 加载 MNIST 模型
async function loadMNIST() {
    mnistModel = await tf.loadLayersModel(
        "https://maths3.github.io/model/model.json"
    );
    console.log("MNIST 模型加载完成");
    mnistReady = true;
}
loadMNIST();

// 清空画布
function clearHandCanvas() {
    const canvas = document.getElementById("handCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// 预处理（适配 Dense 模型）
function preprocessCanvas() {
    const canvas = document.getElementById("handCanvas");

    // 创建 28×28 的临时画布
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tctx = tempCanvas.getContext("2d");

    // 缩放你的手写图像到 28×28
    tctx.drawImage(canvas, 0, 0, 28, 28);

    // 获取像素
    const imgData = tctx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    const input = [];
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // 灰度
        const gray = (r + g + b) / 3;

        // MNIST 是黑底白字 → 必须反色
        const normalized = 1 - gray / 255;

        input.push(normalized);
    }

    // Dense 模型必须 flatten 成 [1, 784]
    return tf.tensor(input).reshape([1, 784]);
}

// 识别数字
async function recognizeDigitMNIST() {
    if (!mnistReady) {
        console.log("MNIST 模型尚未加载");
        return null;
    }

    const input = preprocessCanvas();
    const prediction = mnistModel.predict(input);
    const result = prediction.argMax(1).dataSync()[0];

    return result;
}

// 按钮调用
async function recognizeHandwriting() {
    const digit = await recognizeDigitMNIST();

    if (digit !== null) {
        console.log("MNIST 识别结果：", digit);
        alert("识别结果：" + digit);
    } else {
        alert("模型尚未加载，请稍后再试");
    }
}

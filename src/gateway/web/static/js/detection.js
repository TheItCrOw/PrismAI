// Handle the start of another detection scan
$("body").on("click", "#scan-btn", async function () {
    const document = $(".detection-container .text-input").val();
    if(!document || document === ""){
        alert("You need to set a text to run the detection.")
        return;
    }
    $(".detection-container .loader-div").fadeIn(75);
    try {
        const result = await postNewDetection({ "document": document });
        if (result.status !== 200) {
            alert("An internal error occurred: " + result.payload);
            return;
        }
        console.log(result);

        // Otherwise show the results in the report container
        switchDetectionContainers("report");
        $(".detection-result").html(result.payload.visualization);
        const chartData = {
            "probs": result.payload.probs,
            "labels": result.payload.char_spans
        };
        drawDetectionLinePlot($(".report-container .line-chart")[0], chartData);
        const avg = Number(result.payload.avg).toFixed(2)
        $(".report-container #ai-avg").html(avg)

        const $explainPara = $(".report-container .explanation-text");
        if(avg < 30) $explainPara.html("Overall, the text is very likely human-written.");
        else if(avg < 50) $explainPara.html("Overall, the text is likely human-written, where possibly some bits were AI-generated or written with their assistance.");
        else if(avg < 70) $explainPara.html("Overall, the text is likely AI-generated, but possibly some bits are still human-authored.");
        else if(avg < 90) $explainPara.html("Overall, the text is very likely AI-generated.");
        else $explainPara.html("Overall, the text is very likely completely AI-generated.");
    } catch (ex) {
        alert("An unknown error in the client occured.")
        console.log(ex);
    } finally {
        $(".detection-container .loader-div").hide(0);
    }
})

function switchDetectionContainers(target) {
    if (target === "report") {
        $(".detection-container").addClass("hidden");
        $(".report-container").removeClass("hidden");
        $(".main-content").removeClass("sun-dark");
        $(".main-content").addClass("sun-light");
    } else if (target === "detection") {
        $(".detection-container").removeClass("hidden");
        $(".report-container").addClass("hidden");
        $(".main-content").addClass("sun-dark");
        $(".main-content").removeClass("sun-light");
    }
}

function drawDetectionLinePlot(target, chartData) {
    const lineChart = echarts.init(target);
    const option = {
        grid: {
            top: 10,
            right: 5,
            bottom: 10,
            left: 5,
            containLabel: true
        },
        xAxis: {
            data: chartData.labels,
            axisLabel: { show: false },  // hide labels
            axisTick: { show: false },   // hide ticks
            axisLine: { show: false }    // hide axis line
        },
        yAxis: {
            min: 0,
            max: 100
        },
        visualMap: {
            show: false,
            dimension: 1, // map by value
            min: 0,
            max: 100,
            inRange: {
                color: ['#00cc66', '#ff0000'] // green → red
            }
        },
        series: [
            {
                data: chartData.probs.map((v, i) => [i, v]), // pair index with value
                type: 'line',
                areaStyle: {},
                smooth: true,
                lineStyle: {
                    width: 2
                },
                symbol: 'none'
            }
        ]
    };
    lineChart.setOption(option);
}

async function postNewDetection(detectionDto) {
    const result = await $.ajax({
        type: 'POST',
        contentType: 'application/json',
        url: '/api/detect',
        data: JSON.stringify(detectionDto),
        error: function (error) {
            alert('There was a problem sending the request: ' + error);
            console.log(error);
        }
    });
    return result;
}

$(document).ready(function () {
    console.log("Gateway started.");
    $(".detection-container .loader-div").hide();
})
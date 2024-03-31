
let promises = [
    d3.csv('../airplane.csv', function (d) {
        return {
            ...d,
            Date: d.Year + '-' + d.Month
        }
    }),
    d3.json('../us-states.json')
]
let bar = ''
let baseMap = ''
let lineMap = ''

Promise.all(promises).then((data) => {
    bar = new Bar('barChart', data[0])

    baseMap = new BaseMap('baseMap',data[1],data[0])

    lineMap = new LineMap('lineMapChart',data[1],data[0])
})

const matchKey = {
    "flight": "totalflight",
    "delay": 'DepDelay',
    "cancel": "Cancelled"
}
//bar  categoryBarChange

let barYear = 2016
let barType = 'flight'

function BarChange() {
    barYear = document.getElementById('barYear').value;
    bar.wrangleData()
}

function barTypeChange() {
    barType = document.getElementById('infoBarType').value;
    bar.wrangleData()
}


//line categoryBarChange

let lineYear = 2016
let lineMapType = 'flight'
let isMajorCity=false

function lineMapYearChange() {
    lineYear = document.getElementById('lineMapYear').value;
    lineMap.wrangleData()
}

function LineMapTypeChange() {
    lineMapType = document.getElementById('infoLineMapType').value;
    lineMap.wrangleData()
}

var checkbox = d3.select("#myCheckbox");
checkbox.on("change", function() {
    isMajorCity= checkbox.property("checked");
    lineMap.wrangleData()
});


//BASE MAP 
let baseMapYear = 2016
let baseMapType = 'flight'

function baseMapChange() {
    baseMapYear = document.getElementById('baseMapYear').value;
    baseMap.wrangleData()
}
function baseMapTypeChange() {
    baseMapType = document.getElementById('infoBaseMapType').value;
    baseMap.wrangleData()
}

let scrollTop=0
window.addEventListener("scroll", function() {
    scrollTop = window.scrollY || document.documentElement.scrollTop;

  });
class LineMap {
    constructor(element, geoData, data) {
        this.parentElement = element
        this.geoData = geoData

        this.data = data
        this.lineData = []
        this.majorCity = [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
            "Dallas",
            "San Jose",
            "Austin",
            "Jacksonville",
            "Fort Worth",
            "Columbus",
            "Charlotte",
            "Indianapolis",
            "San Francisco",
            "Seattle",
            "Denver",
            "Nashville",
            "Washington",
            "Oklahoma City",
            "Boston",
            "El Paso",
            "Portland",
            "Las Vegas",
            "Memphis",
            "Detroit",
            "Baltimore",
            "Milwaukee"
        ]

        this.initVis()
    }
    initVis() {
        let vis = this

        vis.margin = {
            top: 50,
            right: 50,
            bottom: 50,
            left: 50
        };
        const w = 1500;
        const h = 1000;
        vis.width = w - vis.margin.left - vis.margin.right;
        vis.height = h - vis.margin.top - vis.margin.bottom;

        vis.svg = d3.select('#' + vis.parentElement).append('svg')
            .attr('width', vis.width + vis.margin.left + vis.margin.right)
            .attr('height', vis.height + vis.margin.top + vis.margin.bottom)
            .attr('viewBox', [0, 0, w, h])


        vis.projection = d3.geoAlbersUsa().translate([vis.width / 2 + vis.margin.left, vis.height / 2])
            .scale([w])

        vis.path = d3.geoPath()
            .projection(vis.projection)

        vis.mapPath = vis.svg.append('g')
            .attr("fill", "#444")
            .attr("cursor", "pointer")
            .selectAll("path")
            .data(this.geoData.features)
            .join("path")
            .attr("stroke", "#fff")
            .attr("d", vis.path)
            .attr('fill', '#ccc');


        //数据范围比例尺  映射到0，1
        vis.scaleLog = d3.scaleSqrt()
            .range([0, 1])


        //线性颜色比例尺  
        vis.colorScale = d3.scaleLinear()
            .domain([0, 0.5, 1])
            .range(["rgba(181, 245, 144, 1)", "rgba(245, 218, 144, 1)", 'rgba(255, 63, 63, 1)']);

        //渐变图例
        const linearGradient = vis.svg.append("defs").append("linearGradient")
            .attr("id", "linearColor")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");


        //有几个颜色  加几个stop  
        linearGradient.append("stop")
            .attr("offset", "0%")
            .style("stop-color", 'rgba(181, 245, 144, 1)');

        linearGradient.append("stop")
            .attr("offset", "50%")
            .style("stop-color", 'rgba(245, 218, 144, 1)');

        linearGradient.append("stop")
            .attr("offset", "100%")
            .style("stop-color", 'rgba(255, 63, 63, 1)');

        vis.legendG = vis.svg.append('g').attr('transform', `translate(${vis.width/2},${vis.height-40})`)

        vis.legend = vis.legendG.append('rect')
            .attr("width", 250)
            .attr("height", 25)
            .style("fill", "url(#" + linearGradient.attr("id") + ")");

        vis.legendScale = d3.scaleLinear()
            .range([0, 250])

        vis.legendAxis = d3.axisBottom()
            .scale(vis.legendScale)


        vis.legendG.append('g')
            .attr('class', 'legend-axis')
            .attr('transform', 'translate(0,25)')

        vis.wrangleData()
    }
    wrangleData() {
        let vis = this
        let filterData = []
        if (lineYear == 'All') {
            filterData = vis.data
        } else {
            filterData = vis.data.filter((d) => {
                return d.Year == lineYear
            })
        }
        filterData = filterData.filter(d => {
            return d.StateD !== d.StateO
        })

        //是否仅展示 重要城市
        if (isMajorCity) {
            filterData = filterData.filter(d => {
                return vis.majorCity.includes(d.CityD) && vis.majorCity.includes(d.CityO)
            })
        }

        const groupData = d3.group(filterData, (d) => d.CityO + '-' + d.CityD)

        const listInfo = Array.from(groupData, ([city, value]) => {
            const total = value.reduce((pre, cur) => {
                return pre + +cur[matchKey[lineMapType]]
            }, 0)
            const flights = value.reduce((pre, cur) => {
                return pre + +cur['totalflight']
            }, 0)
            let chartValue = ''
            if(lineMapType=='flight'){
                chartValue=total
            }else if(lineMapType=='delay'){
                chartValue = (total/flights).toFixed(2)
            }else{
                chartValue = (total/flights *100).toFixed(2)
            }
           
            return {
                source: city.split('-')[0],
                target: city.split('-')[1],
                value: chartValue,
                sourcePosition: [value[0].longitudeO, value[0].latitudeO],
                targetPosition: [value[0].longitudeD, value[0].latitudeD],
                sourceState: stateSym[value[0].StateO],
                targetState: stateSym[value[0].StateD],
            }
        })

        vis.lineData = listInfo

        //准备城市位置数据
        const citiesPosition = listInfo.flatMap(d => [{
                city: d.source,
                position: d.sourcePosition
            },
            {
                city: d.target,
                position: d.targetPosition
            }
        ])

        vis.circleData = Array.from(d3.group(citiesPosition, d => d.city), ([city, value]) => {
            return {
                city,
                position: value[0].position
            }
        })

        vis.updateVis()
    }
    updateVis() {
        let vis = this

        vis.scaleLog.domain(d3.extent(vis.lineData, d => +d.value))

        vis.legendScale.domain(d3.extent(vis.lineData, d => +d.value))
        vis.legendAxis.tickValues(d3.extent(vis.lineData, d => +d.value))
        .tickFormat((d)=>{
            return lineMapType=='cancel'?d+'%':d
        })


        //圆点
        vis.svg.selectAll('.city')
            .data(vis.circleData)
            .join('circle')
            .attr('class', 'city')
            .attr('cx', (d) => {
                return vis.projection(d.position)[0];
            })
            .attr('cy', (d) => {
                return vis.projection(d.position)[1];
            })
            .attr('r', 5)
            .attr('fill', '#136D70')

        //城市名
        vis.svg.selectAll('.cityName')
            .data(vis.circleData)
            .join('text')
            .text(d => d.city)
            .attr('class','cityName')
            .attr('x', (d) => {
                return vis.projection(d.position)[0];
            })
            .attr('y', (d) => {
                return vis.projection(d.position)[1] + 15;
            })
            .attr('text-anchor', 'middle')
            .attr('font-size', '0.6em')

        //航线
        vis.svg.selectAll(".flight")
            .data(vis.lineData)
            .join('path')
            .attr('class', 'flight')
            .attr('d', (d) => {
                const sourceX = vis.projection(d.sourcePosition)[0];
                const sourceY = vis.projection(d.sourcePosition)[1];
                const targetX = vis.projection(d.targetPosition)[0];
                const targetY = vis.projection(d.targetPosition)[1];

                const controlX = (sourceX + targetX) / 2;
                const controlY = sourceX < targetX ? sourceY - 50 : sourceY + 50;

                return "M" + sourceX + "," + sourceY + " Q" + controlX + "," + controlY + " " + targetX + "," + targetY;

            })
            .attr('fill', 'none')
            .attr('stroke', d => {
                //线条  通过比例尺 附上颜色
                return vis.colorScale(vis.scaleLog(d.value))
            })
            .attr('stroke-opacity','0.6')
            .attr('stroke-width', 1.5)
            .on('mouseover', function (event, d) {
                d3.select(this)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', '5')

                let str =
                    `
                <p>${d.source} - ${d.target}</p>
                <p>value: ${lineMapType=='cancel'?d.value+'%':d.value}</p>
                `
                d3.select('.tooltip')
                    .style('opacity', '1')
                    .html(str).style('top', scrollTop + event.y + 20 + 'px')
                    .style('left', event.x + 20 + 'px')
            })
            .on('mouseout', function () {
                d3.select(this)
                    .attr('stroke', d => {
                        //线条  通过比例尺 附上颜色
                        return vis.colorScale(vis.scaleLog(d.value))
                    })
                    .attr('stroke-width', '1')

                d3.select('.tooltip').style('opacity', '0')
            })

        vis.svg.select(".legend-axis").call(vis.legendAxis);
    }
}
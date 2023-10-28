class BaseMap {
    constructor(element, geoData, data) {
        this.parentElement = element
        this.geoData = geoData

        this.data = data


        this.mapData = []

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


        const projection = d3.geoAlbersUsa().translate([vis.width / 2 + vis.margin.left, vis.height / 2])
            .scale([w])

        vis.path = d3.geoPath()
            .projection(projection)

        vis.mapPath = vis.svg.append('g')
            .attr("fill", "#444")
            .attr("cursor", "pointer")
            .selectAll("path")
            .data(this.geoData.features)
            .join("path")
            .attr("stroke", "black")
            .attr("d", vis.path);

        const linearGradient = vis.svg.append("defs").append("linearGradient")
            .attr("id", "baeMapColor")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");

        linearGradient.append("stop")
            .attr("offset", "0%")
            .style("stop-color", '#ffffff');

        linearGradient.append("stop")
            .attr("offset", "100%")
            .style("stop-color", 'rgba(255, 97, 97, 1)');

        vis.legendG = vis.svg.append('g').attr('transform', `translate(${vis.width/2},${vis.height-40})`)

        vis.legend = vis.legendG.append('rect')
            .attr("width", 250)
            .attr("height", 25)
            .style("fill", "url(#" + linearGradient.attr("id") + ")");

        vis.legendScale = d3.scaleLinear()
            .range([0, 250])

        vis.colorScale = d3.scaleSqrt()
            .range(['#ffffff', 'rgba(255, 97, 97, 1)'])

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
        if (baseMapYear == 'All') {
            filterData = vis.data
        } else {
            filterData = vis.data.filter((d) => {
                return d.Year == baseMapYear
            })
        }

        const mergedData = filterData.flatMap(d => [{
                state: d.StateO,
                type: 'departure',
                ...d
            },
            {
                state: d.StateD,
                type: 'arrival',
                ...d
            }
        ]);

        const groupData = d3.group(mergedData, d => d.state)

        const listInfo = Array.from(groupData, ([state, value]) => {
            const total = value.reduce((pre, cur) => {
                return pre + +cur[matchKey[baseMapType]]
            }, 0)

            const flights = value.reduce((pre, cur) => {
                return pre + +cur['totalflight']
            }, 0)

            let chartValue = ''
            if(baseMapType=='flight'){
                chartValue=total
            }else if(baseMapType=='delay'){
                chartValue = (total/flights).toFixed(2)
            }else{
                chartValue = (total/flights *100).toFixed(2)
            }
            return {
                state: stateSym[state],
                value: chartValue
            }
        })


        vis.mapData = listInfo

        vis.updateVis()

    }

    updateVis() {
        let vis = this

        const range = d3.extent(vis.mapData, d => +d.value)

        vis.colorScale.domain(range)

        vis.legendScale.domain(range)
        vis.legendAxis.tickValues(range)

        vis.mapPath.transition()
            .attr('fill', (d) => {

                const temp = vis.mapData.find(item => item.state == d.properties.name)

                return temp ? vis.colorScale(temp.value) : '#fff'
            })

        vis.mapPath.on('mouseover', function (event, d) {
                const temp = vis.mapData.find(item => item.state == d.properties.name)
                if (!temp) return
                d3.select(this)
                    .attr('fill', 'rgba(255,0,0,0.47)')

                let str =
                    `
                    <p>State: ${temp.state}</p>
                    <p>value: ${baseMapType=='cancel'?temp.value+'%':temp.value}</p>
                    `
                d3.select('.tooltip')
                    .style('opacity', '1')
                    .html(str)
            })
            .on('mousemove', (event) => {

                d3.select('.tooltip').style('top', scrollTop + event.y + 20 + 'px')
                    .style('left', event.x + 20 + 'px')
            })
            .on('mouseout', function () {
                d3.select('.tooltip').style('opacity', '0')

                d3.select(this)
                    .attr('fill', (d) => {
                        const temp = vis.mapData.find(item => item.state == d.properties.name)

                        return temp ? vis.colorScale(temp.value) : '#fff'
                    })
            })

        vis.svg.select(".legend-axis").call(vis.legendAxis);
    }
}
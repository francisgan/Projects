class Bar {
    constructor(parentElement, data) {
        this.parentElement = parentElement
        this.data = data

     
        this.barData = []

        this.initVis()
    }
    initVis() {
        let vis = this;

        vis.margin = {
            top: 50,
            right: 50,
            bottom: 50,
            left: 80
        };

        const w = 1000;
        const h = 700;
        vis.width = w - vis.margin.left - vis.margin.right;
        vis.height = h - vis.margin.top - vis.margin.bottom;

        vis.svg = d3.select('#' + vis.parentElement).append('svg')
            .attr('width', vis.width + vis.margin.left + vis.margin.right)
            .attr('height', vis.height + vis.margin.top + vis.margin.bottom)
            .attr('viewBox', [0, 0, w, h])

        vis.container = vis.svg.append('g')
            .attr('transform', `translate (${vis.margin.left}, ${vis.margin.top})`);

        // x y   title
        vis.svg.append('text')
        .text('time')
        .attr('font-size','0.8em')
        .attr('transform',`translate(${vis.width+vis.margin.left+10},${vis.height+vis.margin.top+10})`)
        
        vis.yTitle = vis.svg.append('text')
        .attr('font-size','0.8em')
        .attr('transform',`translate(30,30)`)
     
        
        //tooltip
        vis.tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0)

        //scale

        vis.xScale = d3.scaleBand()
            .range([0, vis.width])
            .padding(0.2)

        vis.yScale = d3.scaleLinear()
            .range([vis.height, 0])


        vis.xAxis = d3.axisBottom(vis.xScale)
        vis.yAxis = d3.axisLeft(vis.yScale)

        vis.container.append('g')
            .attr('transform', `translate(0,${vis.height})`)
            .attr('class', 'x-axis axis')
        vis.container.append('g')
            .attr('class', 'y-axis axis')

        vis.wrangleData();
    }
    wrangleData() {
        let vis = this

        let filterData = []
        if (barYear == 'All') {
            filterData = vis.data
        } else {
            filterData = vis.data.filter((d) => {
                return d.Year == barYear
            })
        }
        const groupData = d3.group(filterData, d => d.Year + '-' + d.Month)
        const listInfo = Array.from(groupData, ([month, value]) => {
            const total = value.reduce((pre, cur) => {
                return pre + +cur[matchKey[barType]]
            }, 0)

            const flights = value.reduce((pre, cur) => {
                return pre + +cur['totalflight']
            }, 0)

            let chartValue = ''
            if(barType=='flight'){
                chartValue=total
            }else if(barType=='delay'){
                chartValue = (total/flights).toFixed(2)
            }else{
                chartValue = (total/flights *100).toFixed(2)
            }
            return {
                month,
                value: chartValue,
          
            }
        })

        vis.barData = listInfo

        vis.updateVis()
    }

    updateVis() {
        let vis = this

        vis.yTitle.text(matchKey[barType])

        vis.xScale.domain(vis.barData.map(d => d.month))
        vis.yScale.domain([0, d3.max(vis.barData, (d) => +d.value)]).nice()

        vis.yAxis.tickFormat(d=>{
            return barType=='cancel'?d+'%':d
        })

        vis.container.selectAll('.myRect')
            .data(vis.barData)
            .join('rect')
            .attr('class', 'myRect')
            .attr('width', vis.xScale.bandwidth())
            .attr('x', (d) => vis.xScale(d.month))
            .attr('y', vis.height)
            .attr('height', (d) => 0)
            .attr('fill', 'rgb(69, 173, 168)')
            .on('mouseover',(event,d)=>{

                let str = 
                `
                <p>Date: ${d.month}</p>
                <p>value: ${barType=='cancel'?d.value +'%':d.value }</p>
                `
                vis.tooltip
                .style('opacity', '1')
                .html(str)
            })
            .on('mousemove',(event)=>{
                vis.tooltip.style('top',  scrollTop + event.y+20 + 'px')
                .style('left', event.x+20 + 'px')
            })
            .on('mouseout',()=>{
                vis.tooltip.style('opacity',0)
            })

        vis.container.selectAll('.myRect').transition(1000)
            .attr('height', (d) => vis.height - vis.yScale(d.value))
            .attr('y', (d) => vis.yScale(d.value))


        vis.svg.select(".y-axis").transition(1000).call(vis.yAxis);
        vis.svg.select(".x-axis").transition(1000).call(vis.xAxis);

        vis.container.selectAll(".x-axis text").style("opacity", (d, i) => {
            if (vis.xScale.domain().length > 15) {
                return i % 2 === 0 ? "1" : "0";
            } else {
                return "1";
            }
        });
    }

}
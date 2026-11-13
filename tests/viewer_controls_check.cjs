const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const source=fs.readFileSync(process.argv[2],'utf8');
new vm.Script(source);
function element(){return {events:{},children:[],classList:{toggle(){}},addEventListener(k,v){this.events[k]=v;},appendChild(x){this.children.push(x);}};}
function run(hasOccupancy) {
 const state={meta:{displayed_point_count:3,point_count:3,cameras:[{id:'front',split:'integration'}]},
   mode_trace_indices:{'0':[0],'1':[1],'2':[2]},camera_indices:hasOccupancy?[4]:[3],default_camera:{eye:{x:1,y:1,z:1}},occupancy:null};
 if (hasOccupancy) {
   state.mode_trace_indices['3']=[3];
   // C-order: [x,y,z], z fastest. Slice Y=0 cells [0,1,4,5].
   state.occupancy={resolution:2,voxel_size_m:.5,origin_m:[10,-3,5],state_base64:Buffer.from([0,1,2,2,2,0,1,1]).toString('base64')};
 }
 const plot={data:Array(hasOccupancy?5:4).fill({})},camera={...element(),checked:false};
 const buttons=[0,1,2,...(hasOccupancy?[3]:[])].map(i=>({...element(),dataset:{mode:String(i)}}));
 const ids={'viewer-data':{textContent:JSON.stringify(state)},'scene-view':plot,'show-cameras':camera,'mode-note':element(),
   'reset-view':element(),'camera-list':element(),'occupancy-controls':hasOccupancy?{...element(),hidden:true}:null,
   'slice-axis':{...element(),value:'1'},'slice-index':{...element(),value:'0'},'slice-label':element()};
 let visibility,geometry,indices,relayout;
 const context={Uint8Array,Map,atob:v=>Buffer.from(v,'base64').toString('binary'),
   document:{getElementById:id=>ids[id],querySelectorAll:()=>buttons,createElement:()=>element()},
   window:{wtumiPlotReady:true,addEventListener(){}},Plotly:{
    restyle(p,o,ii){if(o.visible)visibility=Array.from(o.visible);else{geometry=o;indices=ii;}},
    relayout(p,o){relayout=o;},Plots:{resize(){}}}};
 vm.runInNewContext(source,context);
 buttons[1].events.click();assert.deepEqual(visibility,hasOccupancy?[false,true,false,false,false]:[false,true,false,false]);
 camera.checked=true;camera.events.change();assert.deepEqual(visibility,hasOccupancy?[false,true,false,false,true]:[false,true,false,true]);
 if(hasOccupancy) {
   buttons[3].events.click();assert.deepEqual(visibility,[false,false,false,true,true]);assert.equal(ids['occupancy-controls'].hidden,false);
   assert.deepEqual(Array.from(indices),[3]);
   assert.equal(Math.min(...geometry.x[0]),10);assert.equal(Math.max(...geometry.x[0]),11);
   assert.equal(Math.min(...geometry.y[0]),-3);assert.equal(Math.max(...geometry.y[0]),-2.5);
   assert.equal(Math.min(...geometry.z[0]),5);assert.equal(Math.max(...geometry.z[0]),6);
   assert.equal(geometry.i[0].length,32); // 2x2x1 slab: 16 surface squares, 32 triangles.
   assert.match(ids['slice-label'].textContent,/unknown \/ free \/ occupied = 2 \/ 1 \/ 1/);
   assert.equal(new Set(geometry.facecolor[0]).size,3);
   ids['slice-axis'].value='2';ids['slice-index'].value='1';ids['slice-axis'].events.change();
   assert.equal(Math.min(...geometry.z[0]),5.5);assert.equal(Math.max(...geometry.z[0]),6);
   assert.match(ids['slice-label'].textContent,/unknown \/ free \/ occupied = 1 \/ 2 \/ 1/);
   buttons[0].events.click();assert.equal(ids['occupancy-controls'].hidden,true);
 }
 ids['reset-view'].events.click();assert.equal(JSON.stringify(relayout['scene.camera']),JSON.stringify(state.default_camera));
 assert.equal(ids['camera-list'].children.length,1);
}
run(false);run(true);
console.log('Viewer JavaScript syntax and mocked controls pass: legacy modes, camera isolation, cube slice bounds/colors/axis movement, reset. No browser rendering performed.');

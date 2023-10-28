/* Copyright.

 */
function loginstatus(){
    return(
        api.getPrefs({
        sync: true,
        key:'loginstatus'
    }));
};

function logoutsimple(){
  api.setPrefs({
    key: 'loginstatus',
    value:false
});
api.setPrefs({
    key: 'uuid',
    value:''
});
api.setPrefs({
  key: 'rongtoken',
  value:''
});
api.setPrefs({
key: 'userid',
value:''
});
}

function logout(){
  startloading()
  api.setPrefs({
      key: 'loginstatus',
      value:false
  });
  api.setPrefs({
      key: 'uuid',
      value:''
  });
  api.setPrefs({
    key: 'rongtoken',
    value:''
});
api.setPrefs({
  key: 'userid',
  value:''
});

var rong = api.require('UIRongCloud');
rong.logout(function(ret, err) {});
rong.disconnect({
  isReceivePush: true
}, function(ret, err) {}); 

  setTimeout(function(){
    endloading()
    api.sendEvent({
      name: 'logout',
    });
  },1000)
}


function startloading(){
  var UILoading = api.require('UILoading');
UILoading.flower({
    center: {
        x: api.winWidth/2.0,
        y: api.winHeight/2.0
    },
    size: 40,
    fixed: true
}, function(ret) {
      var loadingid = ret.id;
      api.setGlobalData({
      key: 'loadingid',
      value: loadingid
    });
});
}

function endloading(){
var uiloading = api.require('UILoading');
var loadingid = api.getGlobalData({key:'loadingid'});
uiloading.closeFlower({
    id: loadingid,
    isGradual:false
});
}


function randomString(e) {  
    e = e || 32;
    var t = "ABCDEFGHJKMNPQRSTWXYZabcdefhijkmnprstwxyz0123456789",
    a = t.length,
    n = "";
    for (i = 0; i < e; i++) n += t.charAt(Math.floor(Math.random() * a));
    return n
}

function rongTimeStamp(){
    var date1 = new Date();    //结束时间  
    var date = date1.getTime()
    return date
}

function randomsort(){
  return 0.5 - Math.random()
}

function getgamename(gamecode){
    if(gamecode==1){
        gamename='英雄联盟'
        return gamename;
    }
    else if(gamecode==2){
        gamename='CS:GO'
        return gamename;
    }
    else if(gamecode==3){
      gamename='绝地求生'
      return gamename;
    }
    else if(gamecode==4){
      gamename='云顶之奕'
      return gamename;
    }
    else if(gamecode==5){
      gamename='DOTA'
      return gamename;
    }
    else if(gamecode==6){
      gamename='APEX'
      return gamename;
    }
    else if(gamecode==7){
      gamename='Valorant'
      return gamename;
    }
    else if(gamecode==8){
      gamename='Steam专区'
      return gamename;
    }
    else if(gamecode==9){
      gamename='彩虹六号'
      return gamename;
    }
    else if(gamecode==10){
      gamename='王者荣耀'
      return gamename;
    }
    else if(gamecode==11){
      gamename='LOL手游'
      return gamename;
    }
    else if(gamecode==12){
      gamename='和平精英'
      return gamename;
    }
    else if(gamecode==13){
      gamename='金铲铲之战'
      return gamename;
    }
    else if(gamecode==14){
      gamename='COD手游'
      return gamename;
    }
    else if(gamecode==15){
      gamename='聊天唱歌'
      return gamename;
    }
    else if(gamecode==16){
      gamename='作业指导'
      return gamename;
    }

}

function getgameimgpath(gamecode){
    if(gamecode==1){
        path='../image/lol.png'
        return path;
    }
    else if(gamecode==2){
        path='../image/csgo.jpg'
        return path;
    }
    else if(gamecode==3){
      path='../image/pubg.png'
      return path;
    }
    else if(gamecode==4){
      path='../image/tft.png'
      return path;
    }
    else if(gamecode==5){
      path='../image/dota.jpg'
      return path;
    }
    else if(gamecode==6){
      path='../image/apex.png'
      return path;
    }
    else if(gamecode==7){
      path='../image/valorant.png'
      return path;
    }
    else if(gamecode==8){
      path='../image/steam.png'
      return path;
    }
    else if(gamecode==9){
      path='../image/r6.jpg'
      return path;
    }
    else if(gamecode==10){
      path='../image/wzry.jpg'
      return path;
    }
    else if(gamecode==11){
      path='../image/lolm.png'
      return path;
    }
    else if(gamecode==12){
      path='../image/hpjy.png'
      return path;
    }
    else if(gamecode==13){
      path='../image/tftm.png'
      return path;
    }
    else if(gamecode==14){
      path='../image/cod.png'
      return path;
    }
    else if(gamecode==15){
      path='../image/sing.png'
      return path;
    }
    else if(gamecode==16){
      path='../image/hw.png'
      return path;
    }
}



function encodeUTF8(s) {
    var i, r = [], c, x;
    for (i = 0; i < s.length; i++)
      if ((c = s.charCodeAt(i)) < 0x80) r.push(c);
      else if (c < 0x800) r.push(0xC0 + (c >> 6 & 0x1F), 0x80 + (c & 0x3F));
      else {
        if ((x = c ^ 0xD800) >> 10 == 0) //对四字节UTF-16转换为Unicode
          c = (x << 10) + (s.charCodeAt(++i) ^ 0xDC00) + 0x10000,
            r.push(0xF0 + (c >> 18 & 0x7), 0x80 + (c >> 12 & 0x3F));
        else r.push(0xE0 + (c >> 12 & 0xF));
        r.push(0x80 + (c >> 6 & 0x3F), 0x80 + (c & 0x3F));
      };
    return r;
  }
  
  // 字符串加密成 hex 字符串
  function sha1(s) {
    var data = new Uint8Array(encodeUTF8(s))
    var i, j, t;
    var l = ((data.length + 8) >>> 6 << 4) + 16, s = new Uint8Array(l << 2);
    s.set(new Uint8Array(data.buffer)), s = new Uint32Array(s.buffer);
    for (t = new DataView(s.buffer), i = 0; i < l; i++)s[i] = t.getUint32(i << 2);
    s[data.length >> 2] |= 0x80 << (24 - (data.length & 3) * 8);
    s[l - 1] = data.length << 3;
    var w = [], f = [
      function () { return m[1] & m[2] | ~m[1] & m[3]; },
      function () { return m[1] ^ m[2] ^ m[3]; },
      function () { return m[1] & m[2] | m[1] & m[3] | m[2] & m[3]; },
      function () { return m[1] ^ m[2] ^ m[3]; }
    ], rol = function (n, c) { return n << c | n >>> (32 - c); },
      k = [1518500249, 1859775393, -1894007588, -899497514],
      m = [1732584193, -271733879, null, null, -1009589776];
    m[2] = ~m[0], m[3] = ~m[1];
    for (i = 0; i < s.length; i += 16) {
      var o = m.slice(0);
      for (j = 0; j < 80; j++)
        w[j] = j < 16 ? s[i + j] : rol(w[j - 3] ^ w[j - 8] ^ w[j - 14] ^ w[j - 16], 1),
          t = rol(m[0], 5) + f[j / 20 | 0]() + m[4] + w[j] + k[j / 20 | 0] | 0,
          m[1] = rol(m[1], 30), m.pop(), m.unshift(t);
      for (j = 0; j < 5; j++)m[j] = m[j] + o[j] | 0;
    };
    t = new DataView(new Uint32Array(m).buffer);
    for (var i = 0; i < 5; i++)m[i] = t.getUint32(i << 2);
  
    var hex = Array.prototype.map.call(new Uint8Array(new Uint32Array(m).buffer), function (e) {
      return (e < 16 ? "0" : "") + e.toString(16);
    }).join("");
    return hex;
  }

  function alertnointernet(){
    var dialogBox = api.require('dialogBox');
        dialogBox.scene({
            rect: {
                w: api.winWidth*0.6,                     
                h: 150 
            },
            texts: {
                title: '',
                content: '当前网络不可用，请检查你的网络设置',
                okBtnTitle: '确定'
            },
            styles: {
                bg: 'white',
                maskBg:'rgba(100, 100, 100, 0.5)',
                corner: 20,
                title:{
                    bg: 'white',
                    h: 30,
                    size: 14,
                    color: '#000'
                },
                content:{
                    color: '#000',
                    alignment: 'center',
                    size: 16
                },
                ok: {                             
                h: 45,                         
                bg: '#AAE1DC',                   
                titleColor: '#fff',           
                titleSize: 16                  
    }
                
            },
            tapClose:true,   
        }, function(ret, err) {
            if (ret) {
                dialogBox.close({
                    icon: '',
                    dialogName: 'scene'
                })
            }
        })
  }

function alertservermaintain(){
  var dialogBox = api.require('dialogBox');
        dialogBox.scene({
            rect: {
                w: api.winWidth*0.6,                     
                h: 150 
            },
            texts: {
                title: '',
                content: '服务器发生错误或正在维护',
                okBtnTitle: '确定'
            },
            styles: {
                bg: 'white',
                maskBg:'rgba(100, 100, 100, 0.5)',
                corner: 20,
                title:{
                    bg: 'white',
                    h: 30,
                    size: 14,
                    color: '#000'
                },
                content:{
                    color: '#000',
                    alignment: 'center',
                    size: 16
                },
                ok: {                             
                h: 45,                         
                bg: '#AAE1DC',                   
                titleColor: '#fff',           
                titleSize: 16                  
    }
                
            },
            tapClose:true,   
        }, function(ret, err) {
            if (ret) {
                dialogBox.close({
                    icon: '',
                    dialogName: 'scene'
                })
            }
        })
};

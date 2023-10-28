(function (window) {
  var u = {};
  u.timeago = function (dateTimeStamp, callback) { //如果dateTimeStamp 为13位时间戳则不需要改变，如果是10位则需要补全13位
    var minute = 1000 * 60;
    var hour = minute * 60;
    var day = hour * 24;
    var now = new Date().getTime();
    var diffValue = now - dateTimeStamp;
    if (diffValue < 0) {
      return;
    };
    var minC = diffValue / minute;
    var hourC = diffValue / hour;
    var dayC = diffValue / day;
    if (dayC >= 1 && dayC < 4) {
      result = " " + parseInt(dayC) + "天前"
    } else if (hourC >= 1 && hourC < 24) {
      result = " " + parseInt(hourC) + "小时前"
    } else if (minC >= 1 && minC < 60) {
      result = " " + parseInt(minC) + "分钟前"
    } else if (diffValue >= 0 && diffValue <= minute) {
      result = "1分钟前"
    } else {
      //以上条件都不满足后，用来输出年月日
      var datetime = new Date();
      datetime.setTime(dateTimeStamp);
      var Nyear = datetime.getFullYear();
      var Nmonth = datetime.getMonth() + 1 < 10 ? "0" + (datetime.getMonth() + 1) : datetime.getMonth() + 1;
      var Ndate = datetime.getDate() < 10 ? "0" + datetime.getDate() : datetime.getDate();
      var Nhour = datetime.getHours() < 10 ? "0" + datetime.getHours() : datetime.getHours();
      var Nminute = datetime.getMinutes() < 10 ? "0" + datetime.getMinutes() : datetime.getMinutes();
      result = Nyear + "年" + Nmonth + "月" + Ndate + '日' + ' ' + Nhour + ':' + Nminute
    }
    return result;
  }
  window.$timeNodeConversion = u;

})(window);
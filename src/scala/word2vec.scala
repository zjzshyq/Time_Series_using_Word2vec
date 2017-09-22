/**
  * Created by huyiqing on 2017/8/8 use mmbang.
  * Gone live on 2017/8/14
  * Modefied by vagrant with maven on 2017/8/17
  * Finished and pushed to git on 2017/8/18
  */
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map
import scala.io.Source

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Word2Vec,Word2VecModel}
import org.apache.spark.rdd.RDD
case class SIM(cid: String, simLst: String)

object word2vec{
  def main(args: Array[String]) {
    val params = setParameters(args)
    val sc = setSpark(params)
    val f = sc.sparkContext.textFile(params("files"),params("numPartition").toInt)
    val rdd = f.map(_.split(","))
    if (params("to_predict").toBoolean){
      predict(sc,params,rdd)
    }else{
      trainModel(sc,params,rdd)
    }
  }

  def setSpark(params:Map[String,String]):SparkSession={
    //val master = "local"
    val master = params("master")
    val spark = SparkSession.builder()
      .appName("word2vec").master(master)
      .config("spark.cores.max",params("numCores"))
      .config("spark.executor.memory",params("excutorMemory"))
      .getOrCreate()
    //val conf = new SparkConf().setAppName("word2vec").setMaster(master)
    //val sc = new SparkContext(conf)
    spark
  }

  def trainModel(sc:SparkSession,params:Map[String,String],rdd:RDD[Array[String]]):Word2VecModel={
  
    val user_itemLst_ts = rdd.map(x=>(x(0),List((x(1),x(2).toFloat)))).reduceByKey(_++_,params("numPartition").toInt)
    
    val sorted_itemStr = user_itemLst_ts.map(x=>(x._1,x._2.sortBy(x=>x._2).reverse)).map(x=>x._2.map(y=>y._1).toArray).filter(_.length>1)

    val documentDF = sc.createDataFrame(sorted_itemStr.map(Tuple1.apply)).toDF("cid")
    val word2Vec = new Word2Vec()
      .setInputCol("cid")
      .setOutputCol("result")
      .setVectorSize(params("vectorSize").toInt)
      .setMinCount(params("minCount").toInt)
      .setMaxSentenceLength(params("maxSentenceLength").toInt)
      .setNumPartitions(params("numPartition").toInt)

    val model = word2Vec.fit(documentDF)
    //val result = model.transform(documentDF)
    //result.select("result")
    if (params("save_model").toBoolean){
          model.save("/user/huyiqing/models/word2vecModel_"+getToday())
    }
    model
  }

  def predict(sc:SparkSession,params:Map[String,String],rdd:RDD[Array[String]]){
    var dfLst = new ListBuffer[SIM]

    var i=1
    val model = getModel(sc,params,rdd)

    val items = rdd.map(x=>x(1)).distinct()

    val predict_ts = getTimeStamp()
    items.collect.foreach{
      cno=>
        try{
          val syno = model.findSynonyms(cno,10).collect.map(x=>x(0).toString).toList.mkString(" ")
          dfLst +=SIM(cid=cno,syno)
        }
        catch{case e: IllegalStateException=>i+=1}
    }

    println("the number of words which not in models is "+i.toString)
    val simDf= sc.createDataFrame(dfLst)
    simDf.write.format("csv").save(params("sims"))
  }

  def getToday():String={
    val now = new Date()
    val dateFormat = new SimpleDateFormat("yyyyMMdd")
    dateFormat.format(now)
  }

  def getModel(sc:SparkSession,params:Map[String,String],rdd:RDD[Array[String]]):Word2VecModel={
    if (params("use_model").toBoolean) {
      val model = Word2VecModel.load(params("model"))
      model
    }else{
      val model= trainModel(sc,params,rdd)
      model
    }
  }

  def setParameters(args: Array[String]):Map[String,String]={
    val parmMap = initParamters()
    if (args.length>0) {
      val file=Source.fromFile(args(0))
      for (line<-file.getLines.map(replaceChars)) {
        if (line.matches("^(?!#).*$") && line.length > 1) {
          val keyAndValue = line.split("=")
          if (parmMap.contains(keyAndValue(0)))
            parmMap(keyAndValue(0)) = keyAndValue(1)
        }
      }
      parmMap
    }else {
      println("no dir,use default!")
      System.exit(0)
      parmMap
    }
  }

  def replaceChars(str:String):String = {
    val res = str.replaceAll("(\0|\\s*|\r|\n)", "")
    res
  }

  def initParamters():Map[String,String]={
    var dict:Map[String,String]=Map()
    dict+=("vectorSize"->"30")
    dict+=("minCount"->"1")
    dict+=("maxSentenceLength"->"20")
    dict+=("hdfs"->"hdfs://spark31:9000")
    dict+=("numPartition"->"100")
    dict+=("numCores"->"50")
    dict+=("excutorMemory"->"6g")
    dict+=("model"->"/user/models/20170818")
    dict+=("sims"->"/user/largeData/sims")
    dict+=("save_model"->"true")
    dict+=("use_model"->"true")
    dict+=("to_predict"->"false")
    dict+=("master"->"spark://spark31:7077")
	dict+=("files"->"/user/largeData/mongo/table")
  }
  def getTimeStamp():Float={
    val now = new Date()
    val nowStr = now.getTime().toString
    nowStr.substring(0,nowStr.length-3).toFloat
  }
}


package com.iflytek.scala.ml

import java.io.FileOutputStream

import javax.xml.transform.stream.StreamResult
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.PMMLBuilder

/**
  * 逻辑回归分类
  * 分类算法
  * LR建模
  * * setMaxIter设置最大迭代次数(默认100),具体迭代次数可能在不足最大迭代次数停止
  * * setTol设置容错(默认1e-6),每次迭代会计算一个误差,误差值随着迭代次数增加而减小,当误差小于设置容错,则停止迭代
  * * setRegParam设置正则化项系数(默认0),正则化主要用于防止过拟合现象,如果数据集较小,特征维数又多,易出现过拟合,考虑增大正则化系数
  * * setElasticNetParam正则化范式比(默认0),正则化有两种方式:L1(Lasso)和L2(Ridge),L1用于特征的稀疏化,L2用于防止过拟合
  * * setLabelCol设置标签列
  * * setFeaturesCol设置特征列
  * * setPredictionCol设置预测列
  * * setThreshold设置二分类阈值
  *
  */
object ClassificationLogicTest {

  def main(args: Array[String]): Unit = {

    // 0.构建 Spark 对象
    val spark = SparkSession
      .builder()
      .master("local") // 本地测试，否则报错 A master URL must be set in your configuration at org.apache.spark.SparkContext.
      .appName("test")
      .getOrCreate() // 有就获取无则创建

    import spark.implicits._

    //1 训练样本准备
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3))))).toDF("label", "features")
    training.show(false)

    /**
      * +-----+----------------------------------+
      * |label|features                          |
      * +-----+----------------------------------+
      * |1.0  |(692,[10,20,30],[-1.0,1.5,1.3])   |
      * |0.0  |(692,[45,175,500],[-1.0,1.5,1.3]) |
      * |1.0  |(692,[100,200,300],[-1.0,1.5,1.3])|
      * +-----+----------------------------------+
      */

    //2 建立逻辑回归模型
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    //2 根据训练样本进行模型训练
    val pipeline = new Pipeline().setStages(Array(lr))
    val pipelineModel = pipeline.fit(training)

    //4 测试样本
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3))))).toDF("label", "features")
    test.show(false)

    /**
      * +-----+----------------------------------+
      * |label|features                          |
      * +-----+----------------------------------+
      * |1.0  |(692,[10,20,30],[-1.0,1.5,1.3])   |
      * |0.0  |(692,[45,175,500],[-1.0,1.5,1.3]) |
      * |1.0  |(692,[100,200,300],[-1.0,1.5,1.3])|
      * +-----+----------------------------------+
      */

    //5 对模型进行测试
    val test_predict = pipelineModel.transform(test)
    test_predict
      .select("label", "prediction", "probability", "features")
      .show(false)

    //7 模型保存与加载（发布到服务器 django 时，View 加入如下代码 + 文件）
//    pipelineModel
//      .write
//      .overwrite
//      .save("sparkmlTest/lrmodel")

    val pmml = new PMMLBuilder(test.schema, pipelineModel).build()
    val targetFile = "PMML/pipemodel.pmml"
    val fis = new FileOutputStream(targetFile)
    val fout = new StreamResult(fis)
    JAXBUtil.marshalPMML(pmml, fout)
    println("pmml success......")

  }

}

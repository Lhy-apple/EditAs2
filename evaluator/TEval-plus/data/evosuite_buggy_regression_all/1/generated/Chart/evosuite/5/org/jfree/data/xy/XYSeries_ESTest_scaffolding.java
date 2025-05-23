/**
 * Scaffolding file used to store all the setups needed to run 
 * tests automatically generated by EvoSuite
 * Tue Sep 26 12:56:54 GMT 2023
 */

package org.jfree.data.xy;

import org.evosuite.runtime.annotation.EvoSuiteClassExclude;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.AfterClass;
import org.evosuite.runtime.sandbox.Sandbox;
import org.evosuite.runtime.sandbox.Sandbox.SandboxMode;

@EvoSuiteClassExclude
public class XYSeries_ESTest_scaffolding {

  @org.junit.Rule 
  public org.evosuite.runtime.vnet.NonFunctionalRequirementRule nfr = new org.evosuite.runtime.vnet.NonFunctionalRequirementRule();

  private static final java.util.Properties defaultProperties = (java.util.Properties) java.lang.System.getProperties().clone(); 

  private org.evosuite.runtime.thread.ThreadStopper threadStopper =  new org.evosuite.runtime.thread.ThreadStopper (org.evosuite.runtime.thread.KillSwitchHandler.getInstance(), 3000);


  @BeforeClass 
  public static void initEvoSuiteFramework() { 
    org.evosuite.runtime.RuntimeSettings.className = "org.jfree.data.xy.XYSeries"; 
    org.evosuite.runtime.GuiSupport.initialize(); 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfThreads = 100; 
    org.evosuite.runtime.RuntimeSettings.maxNumberOfIterationsPerLoop = 10000; 
    org.evosuite.runtime.RuntimeSettings.mockSystemIn = true; 
    org.evosuite.runtime.RuntimeSettings.sandboxMode = org.evosuite.runtime.sandbox.Sandbox.SandboxMode.RECOMMENDED; 
    org.evosuite.runtime.sandbox.Sandbox.initializeSecurityManagerForSUT(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.init();
    setSystemProperties();
    initializeClasses();
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
  } 

  @AfterClass 
  public static void clearEvoSuiteFramework(){ 
    Sandbox.resetDefaultSecurityManager(); 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
  } 

  @Before 
  public void initTestCase(){ 
    threadStopper.storeCurrentThreads();
    threadStopper.startRecordingTime();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().initHandler(); 
    org.evosuite.runtime.sandbox.Sandbox.goingToExecuteSUTCode(); 
    setSystemProperties(); 
    org.evosuite.runtime.GuiSupport.setHeadless(); 
    org.evosuite.runtime.Runtime.getInstance().resetRuntime(); 
    org.evosuite.runtime.agent.InstrumentingAgent.activate(); 
  } 

  @After 
  public void doneWithTestCase(){ 
    threadStopper.killAndJoinClientThreads();
    org.evosuite.runtime.jvm.ShutdownHookHandler.getInstance().safeExecuteAddedHooks(); 
    org.evosuite.runtime.classhandling.JDKClassResetter.reset(); 
    resetClasses(); 
    org.evosuite.runtime.sandbox.Sandbox.doneWithExecutingSUTCode(); 
    org.evosuite.runtime.agent.InstrumentingAgent.deactivate(); 
    org.evosuite.runtime.GuiSupport.restoreHeadlessMode(); 
  } 

  public static void setSystemProperties() {
 
    java.lang.System.setProperties((java.util.Properties) defaultProperties.clone()); 
    java.lang.System.setProperty("user.language", "en"); 
    java.lang.System.setProperty("user.country", "US"); 
    java.lang.System.setProperty("user.home", "/home/lhy"); 
    java.lang.System.setProperty("user.dir", "/data/lhy/TEval-plus"); 
    java.lang.System.setProperty("java.io.tmpdir", "/tmp"); 
  }

  private static void initializeClasses() {
    org.evosuite.runtime.classhandling.ClassStateSupport.initializeClasses(XYSeries_ESTest_scaffolding.class.getClassLoader() ,
      "org.jfree.data.general.SeriesChangeListener",
      "org.jfree.data.time.Quarter",
      "org.jfree.data.time.FixedMillisecond",
      "org.jfree.data.xy.XYDataItem",
      "org.jfree.data.time.Year",
      "org.jfree.data.time.SpreadsheetDate",
      "org.jfree.data.statistics.SimpleHistogramBin",
      "org.jfree.data.time.SerialDate",
      "org.jfree.chart.util.ObjectUtilities",
      "org.jfree.chart.util.PublicCloneable",
      "org.jfree.data.time.RegularTimePeriod",
      "org.jfree.data.time.TimePeriodFormatException",
      "org.jfree.data.time.MonthConstants",
      "org.jfree.data.general.Series",
      "org.jfree.data.xy.XYSeries",
      "org.jfree.data.general.SeriesChangeEvent",
      "org.jfree.data.general.SeriesException",
      "org.jfree.data.time.TimeSeriesDataItem",
      "org.jfree.data.time.TimePeriod"
    );
  } 

  private static void resetClasses() {
    org.evosuite.runtime.classhandling.ClassResetter.getInstance().setClassLoader(XYSeries_ESTest_scaffolding.class.getClassLoader()); 

    org.evosuite.runtime.classhandling.ClassStateSupport.resetClasses(
      "org.jfree.data.general.Series",
      "org.jfree.data.xy.XYSeries",
      "org.jfree.data.general.AbstractDataset",
      "org.jfree.data.general.AbstractSeriesDataset",
      "org.jfree.data.xy.AbstractXYDataset",
      "org.jfree.data.xy.AbstractIntervalXYDataset",
      "org.jfree.data.xy.DefaultTableXYDataset",
      "org.jfree.data.general.DatasetGroup",
      "org.jfree.data.xy.IntervalXYDelegate",
      "org.jfree.data.general.DatasetUtilities",
      "org.jfree.data.time.RegularTimePeriod",
      "org.jfree.data.time.Month",
      "org.jfree.data.time.SerialDate",
      "org.jfree.data.xy.XYDataItem",
      "org.jfree.data.time.Day",
      "org.jfree.data.time.Hour",
      "org.jfree.data.time.Year",
      "org.jfree.data.time.TimePeriodFormatException",
      "org.jfree.data.general.SeriesChangeEvent",
      "org.jfree.data.time.SpreadsheetDate",
      "org.jfree.data.time.TimeSeries",
      "org.jfree.data.time.TimeSeriesDataItem",
      "org.jfree.data.ComparableObjectSeries",
      "org.jfree.data.xy.XYIntervalSeries",
      "org.jfree.data.statistics.SimpleHistogramBin",
      "org.jfree.data.time.Second",
      "org.jfree.data.time.Quarter",
      "org.jfree.data.time.Minute",
      "org.jfree.data.time.Millisecond",
      "org.jfree.data.xy.XYIntervalSeriesCollection",
      "org.jfree.data.time.FixedMillisecond",
      "org.jfree.data.time.TimePeriodValues",
      "org.jfree.data.time.TimePeriodValuesCollection",
      "org.jfree.data.time.TimePeriodAnchor",
      "org.jfree.data.general.SeriesException",
      "org.jfree.data.general.WaferMapDataset",
      "org.jfree.data.DefaultKeyedValues2D",
      "org.jfree.chart.util.ObjectUtilities",
      "org.jfree.data.xy.DefaultIntervalXYDataset",
      "org.jfree.data.xy.DefaultWindDataset",
      "org.jfree.data.general.DatasetChangeEvent",
      "org.jfree.data.general.CombinedDataset",
      "org.jfree.data.time.Week",
      "org.jfree.data.xy.XIntervalSeriesCollection",
      "org.jfree.data.time.DynamicTimeSeriesCollection",
      "org.jfree.data.xy.OHLCDataItem",
      "org.jfree.data.time.SimpleTimePeriod",
      "org.jfree.data.time.TimeSeriesCollection",
      "org.jfree.data.statistics.DefaultBoxAndWhiskerXYDataset",
      "org.jfree.data.xy.VectorSeriesCollection",
      "org.jfree.data.general.SubSeriesDataset",
      "org.jfree.data.general.CombinedDataset$DatasetInfo",
      "org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset",
      "org.jfree.data.KeyedObjects2D",
      "org.jfree.data.xy.YIntervalSeriesCollection",
      "org.jfree.data.gantt.TaskSeriesCollection",
      "org.jfree.data.gantt.SlidingGanttCategoryDataset",
      "org.jfree.data.xy.VectorSeries",
      "org.jfree.data.statistics.HistogramDataset",
      "org.jfree.data.statistics.HistogramType",
      "org.jfree.data.statistics.DefaultStatisticalCategoryDataset",
      "org.jfree.data.xy.CategoryTableXYDataset",
      "org.jfree.data.jdbc.JDBCXYDataset",
      "org.jfree.data.gantt.XYTaskDataset",
      "org.jfree.data.xy.XIntervalSeries",
      "org.jfree.data.category.DefaultCategoryDataset",
      "org.jfree.data.jdbc.JDBCCategoryDataset",
      "org.jfree.data.xy.XYBarDataset",
      "org.jfree.data.xy.MatrixSeries",
      "org.jfree.data.category.CategoryToPieDataset",
      "org.jfree.data.statistics.SimpleHistogramDataset",
      "org.jfree.data.xy.DefaultHighLowDataset",
      "org.jfree.data.xy.XYSeriesCollection",
      "org.jfree.data.general.DefaultKeyedValues2DDataset",
      "org.jfree.data.time.TimeTableXYDataset",
      "org.jfree.data.xy.AbstractXYZDataset",
      "org.jfree.data.xy.DefaultXYZDataset",
      "org.jfree.data.xy.DefaultOHLCDataset",
      "org.jfree.data.xy.YIntervalSeries",
      "org.jfree.data.UnknownKeyException",
      "org.jfree.data.general.DefaultPieDataset",
      "org.jfree.data.general.DefaultKeyedValuesDataset",
      "org.jfree.data.DefaultKeyedValues",
      "org.jfree.data.xy.DefaultXYDataset"
    );
  }
}

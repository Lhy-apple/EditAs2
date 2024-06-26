/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:19:33 GMT 2023
 */

package org.jfree.chart.plot;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.Color;
import java.awt.Paint;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartRenderingInfo;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.axis.PeriodAxisLabelInfo;
import org.jfree.chart.plot.MultiplePiePlot;
import org.jfree.chart.util.TableOrder;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultIntervalCategoryDataset;
import org.jfree.data.general.DefaultKeyedValues2DDataset;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import org.jfree.data.statistics.DefaultMultiValueCategoryDataset;
import org.jfree.data.statistics.DefaultStatisticalCategoryDataset;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultiplePiePlot_ESTest extends MultiplePiePlot_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultKeyedValues2DDataset defaultKeyedValues2DDataset0 = new DefaultKeyedValues2DDataset();
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultKeyedValues2DDataset0);
      String string0 = multiplePiePlot0.getPlotType();
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
      assertEquals("Multiple Pie Plot", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      Comparable comparable0 = multiplePiePlot0.getAggregatedItemsKey();
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
      assertEquals("Other", comparable0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[][] doubleArray0 = new double[3][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      multiplePiePlot0.setLimit(1.0625);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      jFreeChart0.createBufferedImage(10, 10);
      assertEquals(1.0625, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      double double0 = multiplePiePlot0.getLimit();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      multiplePiePlot0.getDataExtractOrder();
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[][] doubleArray0 = new double[6][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      Color color0 = (Color)multiplePiePlot0.getAggregatedItemsPaint();
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
      assertEquals(192, color0.getBlue());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot((CategoryDataset) null);
      multiplePiePlot0.setDataset((CategoryDataset) null);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultStatisticalCategoryDataset0);
      multiplePiePlot0.setDataset(defaultStatisticalCategoryDataset0);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      JFreeChart jFreeChart0 = multiplePiePlot0.getPieChart();
      multiplePiePlot0.setPieChart(jFreeChart0);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultMultiValueCategoryDataset defaultMultiValueCategoryDataset0 = new DefaultMultiValueCategoryDataset();
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultMultiValueCategoryDataset0);
      // Undeclared exception!
      try { 
        multiplePiePlot0.setPieChart((JFreeChart) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'pieChart' argument.
         //
         verifyException("org.jfree.chart.plot.MultiplePiePlot", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultBoxAndWhiskerCategoryDataset0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      // Undeclared exception!
      try { 
        multiplePiePlot0.setPieChart(jFreeChart0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The 'pieChart' argument must be a chart based on a PiePlot.
         //
         verifyException("org.jfree.chart.plot.MultiplePiePlot", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[][] doubleArray0 = new double[6][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      TableOrder tableOrder0 = TableOrder.BY_ROW;
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      multiplePiePlot0.setDataExtractOrder(tableOrder0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      jFreeChart0.createBufferedImage(10, 12);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultKeyedValues2DDataset defaultKeyedValues2DDataset0 = new DefaultKeyedValues2DDataset();
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultKeyedValues2DDataset0);
      // Undeclared exception!
      try { 
        multiplePiePlot0.setDataExtractOrder((TableOrder) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'order' argument
         //
         verifyException("org.jfree.chart.plot.MultiplePiePlot", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      // Undeclared exception!
      try { 
        multiplePiePlot0.setAggregatedItemsKey((Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'key' argument.
         //
         verifyException("org.jfree.chart.plot.MultiplePiePlot", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      Color color0 = (Color)PeriodAxisLabelInfo.DEFAULT_DIVIDER_PAINT;
      multiplePiePlot0.setAggregatedItemsPaint(color0);
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot();
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot1);
      assertFalse(boolean0);
      assertEquals(0.0, multiplePiePlot1.getLimit(), 0.01);
      assertFalse(multiplePiePlot1.equals((Object)multiplePiePlot0));
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      // Undeclared exception!
      try { 
        multiplePiePlot0.setAggregatedItemsPaint((Paint) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'paint' argument.
         //
         verifyException("org.jfree.chart.plot.MultiplePiePlot", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultMultiValueCategoryDataset defaultMultiValueCategoryDataset0 = new DefaultMultiValueCategoryDataset();
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultMultiValueCategoryDataset0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      jFreeChart0.createBufferedImage(10, 10);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[][] doubleArray0 = new double[3][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      TableOrder tableOrder0 = TableOrder.BY_ROW;
      multiplePiePlot0.setDataExtractOrder(tableOrder0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      jFreeChart0.createBufferedImage(10, 10);
      LegendItemCollection legendItemCollection0 = multiplePiePlot0.getLegendItems();
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
      assertEquals(5, legendItemCollection0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[][] doubleArray0 = new double[2][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      jFreeChart0.createBufferedImage(200, 300);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[][] doubleArray0 = new double[16][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      ChartPanel chartPanel0 = new ChartPanel(jFreeChart0, false);
      ChartRenderingInfo chartRenderingInfo0 = chartPanel0.getChartRenderingInfo();
      jFreeChart0.createBufferedImage(10, 10, chartRenderingInfo0);
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot1);
      assertEquals(0.0, multiplePiePlot1.getLimit(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[][] doubleArray0 = new double[23][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      JFreeChart jFreeChart0 = new JFreeChart(multiplePiePlot0);
      jFreeChart0.createBufferedImage(10, 10);
      LegendItemCollection legendItemCollection0 = multiplePiePlot0.getLegendItems();
      assertEquals(23, legendItemCollection0.getItemCount());
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      multiplePiePlot0.getLegendItems();
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[][] doubleArray0 = new double[16][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot(defaultIntervalCategoryDataset0);
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot1);
      assertTrue(boolean0);
      assertEquals(0.0, multiplePiePlot1.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot0);
      assertTrue(boolean0);
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      boolean boolean0 = multiplePiePlot0.equals("3XpD}lv)]");
      assertEquals(0.0, multiplePiePlot0.getLimit(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot();
      TableOrder tableOrder0 = TableOrder.BY_ROW;
      multiplePiePlot1.setDataExtractOrder(tableOrder0);
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot1);
      assertFalse(multiplePiePlot1.equals((Object)multiplePiePlot0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot();
      assertEquals(0.0, multiplePiePlot1.getLimit(), 0.01);
      
      multiplePiePlot1.setLimit(1.0F);
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot();
      assertTrue(multiplePiePlot1.equals((Object)multiplePiePlot0));
      
      multiplePiePlot1.setAggregatedItemsKey(10);
      boolean boolean0 = multiplePiePlot0.equals(multiplePiePlot1);
      assertFalse(multiplePiePlot1.equals((Object)multiplePiePlot0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MultiplePiePlot multiplePiePlot0 = new MultiplePiePlot();
      MultiplePiePlot multiplePiePlot1 = new MultiplePiePlot();
      assertTrue(multiplePiePlot1.equals((Object)multiplePiePlot0));
      
      multiplePiePlot1.setNoDataMessagePaint(multiplePiePlot0.DEFAULT_BACKGROUND_PAINT);
      boolean boolean0 = multiplePiePlot1.equals(multiplePiePlot0);
      assertFalse(boolean0);
  }
}

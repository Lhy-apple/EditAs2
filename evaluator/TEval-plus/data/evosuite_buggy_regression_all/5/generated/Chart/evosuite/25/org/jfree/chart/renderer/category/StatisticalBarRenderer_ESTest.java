/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:26:43 GMT 2023
 */

package org.jfree.chart.renderer.category;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import javax.swing.text.DefaultCaret;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.axis.CategoryAxis3D;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis3D;
import org.jfree.chart.axis.SubCategoryAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.CombinedRangeCategoryPlot;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.chart.renderer.category.CategoryItemRendererState;
import org.jfree.chart.renderer.category.CategoryStepRenderer;
import org.jfree.chart.renderer.category.StatisticalBarRenderer;
import org.jfree.chart.renderer.category.WaterfallBarRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import org.jfree.data.statistics.DefaultStatisticalCategoryDataset;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StatisticalBarRenderer_ESTest extends StatisticalBarRenderer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Color color0 = (Color)statisticalBarRenderer0.getErrorIndicatorPaint();
      assertEquals((-8355712), color0.getRGB());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      BasicStroke basicStroke0 = (BasicStroke)statisticalBarRenderer0.getErrorIndicatorStroke();
      assertEquals(0.5F, basicStroke0.getLineWidth(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      statisticalBarRenderer0.setErrorIndicatorStroke(statisticalBarRenderer0.DEFAULT_OUTLINE_STROKE);
      assertEquals(2.0, statisticalBarRenderer0.getItemLabelAnchorOffset(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryStepRenderer.State categoryStepRenderer_State0 = new CategoryStepRenderer.State((PlotRenderingInfo) null);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawItem((Graphics2D) null, categoryStepRenderer_State0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultBoxAndWhiskerCategoryDataset0, 500, 10, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires StatisticalCategoryDataset.
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 0, (Comparable) 0, (Comparable) false);
      defaultStatisticalCategoryDataset0.add((Number) 0, (Number) 3.0, (Comparable) 3.0, (Comparable) 1.0E-8);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D((String) null);
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot0, 0, (PlotRenderingInfo) null);
      SubCategoryAxis subCategoryAxis0 = new SubCategoryAxis((String) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, subCategoryAxis0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CategoryPlot categoryPlot0 = new CategoryPlot();
      CategoryStepRenderer.State categoryStepRenderer_State0 = new CategoryStepRenderer.State((PlotRenderingInfo) null);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      Byte byte0 = new Byte((byte) (-73));
      defaultStatisticalCategoryDataset0.add((Number) byte0, (Number) 0.2, (Comparable) (byte) (-73), (Comparable) true);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryStepRenderer_State0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryItemRendererState categoryItemRendererState0 = new CategoryItemRendererState((PlotRenderingInfo) null);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 0.05, (Number) 0, (Comparable) 1.0F, (Comparable) false);
      NumberAxis3D numberAxis3D0 = new NumberAxis3D();
      statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, numberAxis3D0, defaultStatisticalCategoryDataset0, 0, 0);
      assertNull(numberAxis3D0.getLabelURL());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 0.05, (Number) 0, (Comparable) 2.0F, (Comparable) 3.0);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, combinedRangeCategoryPlot0, 247, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 0.05, (Comparable) 0.0F, (Comparable) true);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, combinedRangeCategoryPlot0, 0, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add(combinedRangeCategoryPlot0.ZERO, (Number) statisticalBarRenderer0.ZERO, (Comparable) 0.2, (Comparable) 0);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, combinedRangeCategoryPlot0, 500, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot0, 1, (PlotRenderingInfo) null);
      SubCategoryAxis subCategoryAxis0 = new SubCategoryAxis((String) null);
      defaultStatisticalCategoryDataset0.add((Number) 0.0F, (Number) 1.0E-8, (Comparable) 2.0F, (Comparable) true);
      statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, subCategoryAxis0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
      assertFalse(subCategoryAxis0.isTickMarksVisible());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot0, (-1), (PlotRenderingInfo) null);
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 10, (Comparable) 2.0F, (Comparable) 0);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CategoryPlot categoryPlot0 = new CategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 10, (Comparable) 0, (Comparable) 0.05);
      defaultStatisticalCategoryDataset0.add((Number) 1.0F, (Number) 0, (Comparable) 2.0F, (Comparable) 0);
      CategoryPlot categoryPlot1 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot1, 0, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawVerticalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, (-1701), 500);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 0.2, (Comparable) null, (Comparable) 10);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot0, 10, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0, 10);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CategoryPlot categoryPlot0 = new CategoryPlot();
      CategoryStepRenderer.State categoryStepRenderer_State0 = new CategoryStepRenderer.State((PlotRenderingInfo) null);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      Byte byte0 = new Byte((byte) (-73));
      defaultStatisticalCategoryDataset0.add((Number) byte0, (Number) 0.2, (Comparable) (byte) (-73), (Comparable) true);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawItem((Graphics2D) null, categoryStepRenderer_State0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CategoryPlot categoryPlot0 = new CategoryPlot();
      CategoryItemRendererState categoryItemRendererState0 = new CategoryItemRendererState((PlotRenderingInfo) null);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 0, (Number) 2.0F, (Comparable) 0.2, (Comparable) 0.2);
      statisticalBarRenderer0.drawVerticalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
      assertEquals(0.0, statisticalBarRenderer0.getLowerClip(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 0.0F, (Number) 0, (Comparable) true, (Comparable) 2.0F);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, combinedRangeCategoryPlot0, 0, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawVerticalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 0.05, (Comparable) 0.0F, (Comparable) true);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, combinedRangeCategoryPlot0, 0, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawVerticalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      defaultStatisticalCategoryDataset0.add((Number) 0.05, (Number) 1.0E-8, (Comparable) 2.0F, (Comparable) 1.0F);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, combinedRangeCategoryPlot0, 10, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawVerticalItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      defaultStatisticalCategoryDataset0.add((Number) 0.0F, (Number) 1.0F, (Comparable) 0.0F, (Comparable) true);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot0, 1931656145, (PlotRenderingInfo) null);
      statisticalBarRenderer0.drawItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0, (-1));
      assertEquals(0.05, ValueAxis.DEFAULT_LOWER_MARGIN, 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultCaret defaultCaret0 = new DefaultCaret();
      LogAxis logAxis0 = new LogAxis();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultStatisticalCategoryDataset0, categoryAxis3D0, logAxis0, statisticalBarRenderer0);
      WaterfallBarRenderer waterfallBarRenderer0 = new WaterfallBarRenderer();
      CategoryPlot categoryPlot1 = new CategoryPlot((CategoryDataset) null, categoryAxis3D0, logAxis0, waterfallBarRenderer0);
      defaultStatisticalCategoryDataset0.add((Number) 10, (Number) 0.2, (Comparable) null, (Comparable) 10);
      CategoryItemRendererState categoryItemRendererState0 = statisticalBarRenderer0.initialise((Graphics2D) null, defaultCaret0, categoryPlot0, 10, (PlotRenderingInfo) null);
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawItem((Graphics2D) null, categoryItemRendererState0, defaultCaret0, categoryPlot0, categoryAxis3D0, logAxis0, defaultStatisticalCategoryDataset0, 0, 0, 10);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Object object0 = statisticalBarRenderer0.clone();
      boolean boolean0 = statisticalBarRenderer0.equals(object0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      boolean boolean0 = statisticalBarRenderer0.equals(statisticalBarRenderer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      boolean boolean0 = statisticalBarRenderer0.equals(defaultStatisticalCategoryDataset0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      StatisticalBarRenderer statisticalBarRenderer1 = (StatisticalBarRenderer)statisticalBarRenderer0.clone();
      assertTrue(statisticalBarRenderer1.equals((Object)statisticalBarRenderer0));
      
      statisticalBarRenderer1.setItemMargin(3.0);
      boolean boolean0 = statisticalBarRenderer0.equals(statisticalBarRenderer1);
      assertFalse(statisticalBarRenderer1.equals((Object)statisticalBarRenderer0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      StatisticalBarRenderer statisticalBarRenderer1 = new StatisticalBarRenderer();
      assertTrue(statisticalBarRenderer1.equals((Object)statisticalBarRenderer0));
      
      statisticalBarRenderer1.setErrorIndicatorPaint(statisticalBarRenderer0.DEFAULT_VALUE_LABEL_PAINT);
      boolean boolean0 = statisticalBarRenderer0.equals(statisticalBarRenderer1);
      assertFalse(statisticalBarRenderer1.equals((Object)statisticalBarRenderer0));
      assertFalse(boolean0);
  }
}

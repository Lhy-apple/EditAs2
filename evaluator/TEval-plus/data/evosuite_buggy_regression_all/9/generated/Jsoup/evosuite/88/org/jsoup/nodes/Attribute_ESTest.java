/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:16:53 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attribute;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Attribute_ESTest extends Attribute_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Attribute attribute0 = new Attribute("hidde", "hidde");
      boolean boolean0 = attribute0.isDataAttribute();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Attribute attribute0 = new Attribute("data-data-)Ep", (String) null);
      Attribute attribute1 = attribute0.clone();
      boolean boolean0 = attribute0.equals(attribute1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("nohref", "nohref");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      boolean boolean0 = attribute0.shouldCollapseAttribute(document_OutputSettings0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("1Tl{H8in", "1Tl{H8in");
      attribute0.setKey("1Tl{H8in");
      assertEquals("1Tl{H8in", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("=L9wa", "=L9wa", attributes0);
      attribute0.setKey("=L9wa");
      assertEquals("=L9wa", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("=L9wa", "=L9wa", attributes0);
      attributes0.put(attribute0);
      attribute0.setKey("=L9wa");
      assertEquals("=L9wa", attribute0.getKey());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("=L9wa", "=L9wa", attributes0);
      attributes0.put(attribute0);
      String string0 = attribute0.setValue("=L9wa");
      assertEquals("=L9wa", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attribute attribute0 = new Attribute("data-data-)Ep", (String) null);
      String string0 = attribute0.html();
      assertEquals("data-data-)Ep", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      boolean boolean0 = Attribute.isDataAttribute("data-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      boolean boolean0 = Attribute.isDataAttribute("data-2W m\"lXTR!gGPBm");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      boolean boolean0 = Attribute.shouldCollapseAttribute((String) null, (String) null, document_OutputSettings0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attribute attribute0 = new Attribute("UKF+", "");
      String string0 = attribute0.toString();
      assertEquals("UKF+=\"\"", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attribute attribute0 = new Attribute(".3cS8W", "Lb(9a[y;O");
      String string0 = attribute0.toString();
      assertEquals(".3cS8W=\"Lb(9a[y;O\"", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("nowrap", "nowrap");
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("?6%S%qZX", "?6%S%qZX", attributes0);
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("?6%S%qZX", "?6%S%qZX", attributes0);
      String string0 = attribute0.setValue((String) null);
      assertNotNull(string0);
      
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("?6%S%qZX", "?6%S%qZX");
      Attribute attribute1 = Attribute.createFromEncoded("?6%S%qZX=\"?6%S%qZX\"", "?6%S%qZX=\"?6%S%qZX\"");
      boolean boolean0 = attribute0.equals(attribute1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attribute attribute0 = new Attribute("nowrap", "nowrap");
      boolean boolean0 = attribute0.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("nowrap", "nowrap");
      boolean boolean0 = attribute0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attribute attribute0 = new Attribute("nowrap", "nowrap");
      boolean boolean0 = attribute0.equals("nowrap");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attribute attribute0 = new Attribute("nowrap", "nowrap");
      Attribute attribute1 = attribute0.clone();
      boolean boolean0 = attribute1.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("2W \"lTRE!PXm", "2W \"lTRE!PXm");
      Attribute attribute1 = new Attribute("2W \"lTRE!PXm", (String) null);
      boolean boolean0 = attribute1.equals(attribute0);
      assertFalse(boolean0);
      assertTrue(attribute0.equals((Object)attribute1));
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("?6%S%qZX", "?6%S%qZX");
      attribute0.hashCode();
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attribute attribute0 = new Attribute("selected", (String) null);
      attribute0.hashCode();
  }
}
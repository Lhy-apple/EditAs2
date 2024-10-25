/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:29:58 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.CharArrayWriter;
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
      Attribute attribute0 = Attribute.createFromEncoded("9 ", "9 ");
      attribute0.html();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("(Vm}", "(Vm}");
      Attributes attributes0 = new Attributes();
      attributes0.put(attribute0);
      attribute0.setKey("(Vm}");
      assertEquals("(Vm}", attribute0.getKey());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attribute attribute0 = new Attribute("reversed", "reversed");
      boolean boolean0 = attribute0.isDataAttribute();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attribute attribute0 = new Attribute("M=C{'U", (String) null);
      Attribute attribute1 = attribute0.clone();
      boolean boolean0 = attribute1.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("^XF\u0000W6oemh}\"C", "^XF\u0000W6oemh}\"C");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      boolean boolean0 = attribute0.shouldCollapseAttribute(document_OutputSettings0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("CG(lqJeI@+", "CG(lqJeI@+");
      attribute0.setKey("CG(lqJeI@+");
      assertEquals("CG(lqJeI@+", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("y]2G", "y]2G", attributes0);
      attribute0.setKey("y]2G");
      assertEquals("y]2G", attribute0.getKey());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("org.jsoup.SerializationException", "org.jsoup.SerializationException", attributes0);
      attribute0.setValue("AfterDoctypePublicIdentifier");
      assertEquals("AfterDoctypePublicIdentifier", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("M=C{'U", "M=C{'U");
      Attribute attribute0 = new Attribute("M=C{'U", "M=C{'U", attributes0);
      String string0 = attribute0.setValue("M=C{'U");
      assertEquals("M=C{'U", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attribute attribute0 = new Attribute("compact", "compact");
      String string0 = attribute0.toString();
      assertEquals("compact", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      boolean boolean0 = Attribute.isDataAttribute("data-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      boolean boolean0 = Attribute.isDataAttribute("data-d:");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("mul_tiple", "mul_tiple");
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      attribute0.html((Appendable) charArrayWriter0, document_OutputSettings0);
      assertEquals(21, charArrayWriter0.size());
      assertEquals("mul_tiple=\"mul_tiple\"", charArrayWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Attribute attribute0 = new Attribute("Fj-xS]|,BM]f[Q'/9;", (String) null);
      String string0 = attribute0.toString();
      assertEquals("Fj-xS]|,BM]f[Q'/9;", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("xD.kv5#&<+/ivxui", "", attributes0);
      String string0 = attribute0.toString();
      assertEquals("xD.kv5#&<+/ivxui=\"\"", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("compact", "compact");
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("data-", "data-");
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attribute attribute0 = new Attribute("%MG)", (String) null);
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attribute attribute0 = new Attribute("compact", "compact");
      boolean boolean0 = attribute0.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("CG(lqJeI@+", "CG(lqJeI@+");
      boolean boolean0 = attribute0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("data-", "data-");
      boolean boolean0 = attribute0.equals("data-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attribute attribute0 = new Attribute("96E:E_9C4WdE", "96E:E_9C4WdE");
      Attribute attribute1 = new Attribute("<", "<");
      boolean boolean0 = attribute1.equals(attribute0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attribute attribute0 = new Attribute("compact", "compact");
      Attribute attribute1 = attribute0.clone();
      boolean boolean0 = attribute0.equals(attribute1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attribute attribute0 = new Attribute("JCVLar b!*q=", "JCVLar b!*q=");
      Attribute attribute1 = new Attribute("JCVLar b!*q=", (String) null);
      boolean boolean0 = attribute1.equals(attribute0);
      assertFalse(boolean0);
      assertTrue(attribute0.equals((Object)attribute1));
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("9 ", "9 ");
      attribute0.hashCode();
      assertEquals("9 ", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attribute attribute0 = new Attribute("Fj-xS]|,BM]f[Q'/9;", (String) null);
      attribute0.hashCode();
  }
}

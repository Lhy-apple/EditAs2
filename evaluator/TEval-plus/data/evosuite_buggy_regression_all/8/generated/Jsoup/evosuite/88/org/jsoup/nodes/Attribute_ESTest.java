/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:38:42 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      Attribute attribute0 = Attribute.createFromEncoded("checked", "checked");
      attribute0.html();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Attribute attribute0 = new Attribute("data-", "data-");
      String string0 = attribute0.getKey();
      assertEquals("data-", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("ch$c%ked", "ch$c%ked");
      boolean boolean0 = attribute0.isDataAttribute();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Attribute attribute0 = new Attribute(">U>_?7", (String) null);
      Attribute attribute1 = attribute0.clone();
      boolean boolean0 = attribute1.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Attribute attribute0 = new Attribute("(25'O?](-%", "(25'O?](-%");
      String string0 = attribute0.getValue();
      assertEquals("(25'O?](-%", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("data-U3zMszgF'ZAbe.9J0", "");
      String string0 = attribute0.toString();
      assertEquals("data-U3zMszgF'ZAbe.9J0=\"\"", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("i88&V=", "i88&V=");
      // Undeclared exception!
      try { 
        attribute0.shouldCollapseAttribute((Document.OutputSettings) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Attribute", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Attribute attribute0 = new Attribute(">", ">");
      attribute0.setKey(">");
      assertEquals(">", attribute0.getKey());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("_R&>#gY}Uh1mJD]", "_R&>#gY}Uh1mJD]", attributes0);
      attribute0.setKey("Azl]~ke");
      assertEquals("Azl]~ke", attribute0.getKey());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("zzc+yvrIXR", "zzc+yvrIXR");
      Attribute attribute0 = new Attribute("zzc+yvrIXR", "zzc+yvrIXR", attributes0);
      attribute0.setKey("zzc+yvrIXR");
      assertEquals("zzc+yvrIXR", attribute0.getKey());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Attribute attribute0 = new Attribute("ag name must not be empty.", "ag name must not be empty.", attributes0);
      String string0 = attribute0.setValue("ag name must not be empty.");
      assertEquals("", string0);
      assertEquals("ag name must not be empty.", attribute0.getValue());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      attributes0.put("CSqu", "CSqu");
      Attribute attribute0 = new Attribute("CSqu", "CSqu", attributes0);
      String string0 = attribute0.setValue("CSqu");
      assertEquals("CSqu", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      boolean boolean0 = Attribute.isDataAttribute("data-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = Attribute.isDataAttribute("data-CMV");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      boolean boolean0 = Attribute.shouldCollapseAttribute("", "", document_OutputSettings0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Attribute attribute0 = new Attribute("tr", (String) null);
      String string0 = attribute0.toString();
      assertEquals("tr", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Attribute attribute0 = new Attribute("m(=hh:5i77Z;`.5`N^", "data-compact");
      String string0 = attribute0.toString();
      assertEquals("m(=hh:5i77Z;`.5`N^=\"data-compact\"", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Attribute attribute0 = new Attribute("itemscope", "itemscope");
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Attribute attribute0 = new Attribute("Tag name must not be empty.", "Tag name must not be empty.");
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Attribute attribute0 = new Attribute("Tag name must not be empty.", (String) null);
      boolean boolean0 = attribute0.isBooleanAttribute();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("Tg name must not be empty.", "Tg name must not be empty.");
      boolean boolean0 = attribute0.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("_R&>#Y}Uh1JD]", "_R&>#Y}Uh1JD]");
      boolean boolean0 = attribute0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("Tag name must not be empty.", "Tag name must not be empty.");
      boolean boolean0 = attribute0.equals("Tag name must not be empty.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Attribute attribute0 = new Attribute("ins", "Tag name must not be empty.");
      Attribute attribute1 = Attribute.createFromEncoded("Tag name must not be empty.", "ins");
      boolean boolean0 = attribute1.equals(attribute0);
      assertEquals("ins", attribute1.getValue());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Attribute attribute0 = new Attribute("Tag name must not be empty.", "Tag name must not be empty.");
      Attribute attribute1 = new Attribute("Tag name must not be empty.", "Tag name must not be empty.");
      boolean boolean0 = attribute1.equals(attribute0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("Tag name must not be empty.", "Tag name must not be empty.");
      Attribute attribute1 = new Attribute("Tag name must not be empty.", (String) null);
      boolean boolean0 = attribute1.equals(attribute0);
      assertFalse(boolean0);
      assertTrue(attribute0.equals((Object)attribute1));
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Attribute attribute0 = Attribute.createFromEncoded("checked", "checked");
      attribute0.hashCode();
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Attribute attribute0 = new Attribute("Tag name must not be empty.", (String) null);
      attribute0.hashCode();
  }
}
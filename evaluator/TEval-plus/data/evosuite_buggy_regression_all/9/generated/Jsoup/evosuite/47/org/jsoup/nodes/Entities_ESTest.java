/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:09:18 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String string0 = Entities.unescape("&nbsp;");
      assertEquals("\u00A0", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("suchthat");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("gt");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      boolean boolean0 = Entities.isBaseNamedEntity("g");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) " Gqjh2xfRHnB<0");
      Entities.escape(stringBuilder0, " Gqjh2xfRHnB<0", document_OutputSettings0, true, true, true);
      assertEquals(" Gqjh2xfRHnB<0Gqjh2xfRHnB<0", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "[\"0e` r)& i#", document_OutputSettings0, true, true, true);
      assertEquals("[&quot;0e` r)&amp; i#", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder("noscript");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Entities.escape(stringBuilder0, "   ", document_OutputSettings0, false, true, false);
      assertEquals("noscript ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder(">]`[4dZ");
      Entities.escape(stringBuilder0, ">]`[4dZ", document_OutputSettings0, true, true, true);
      assertEquals(">]`[4dZ>]`[4dZ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("\u00A0", document_OutputSettings0);
      assertEquals("&nbsp;", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Entities.EscapeMode entities_EscapeMode0 = Entities.EscapeMode.xhtml;
      document_OutputSettings0.escapeMode(entities_EscapeMode0);
      String string0 = Entities.escape("\u00A0", document_OutputSettings0);
      assertEquals("&#xa0;", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape(" Gqjh2xfRHnB<06", document_OutputSettings0);
      assertEquals(" Gqjh2xfRHnB&lt;06", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("MNjprn>nG", document_OutputSettings0);
      assertEquals("MNjprn&gt;nG", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("\"w^@G", document_OutputSettings0);
      assertEquals("\"w^@G", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.charset("US-ASCII");
      String string0 = Entities.escape("\u00A3", document_OutputSettings0);
      assertEquals("&pound;", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.charset("ascii");
      String string0 = Entities.escape("\u0ADE", document_OutputSettings0);
      assertEquals("&#xade;", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.charset("ascii");
      String string0 = Entities.escape("ascii", document_OutputSettings1);
      assertEquals("ascii", string0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:11:16 GMT 2023
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
      String string0 = Entities.unescape("&amp;");
      assertEquals("&", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String string0 = Entities.unescape("4&+^cIx1ZGqMss=.}8");
      assertEquals("4&+^cIx1ZGqMss=.}8", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      boolean boolean0 = Entities.isNamedEntity("quot");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "jx6+ 3fW>7a1jLo", document_OutputSettings0, true, true, true);
      assertEquals("jx6+ 3fW>7a1jLo", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder(96);
      Entities.escape(stringBuilder0, "   ", document_OutputSettings0, false, true, true);
      assertEquals("", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder(96);
      Entities.escape(stringBuilder0, "   ", document_OutputSettings0, false, true, false);
      assertEquals(" ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "N}\"wJ\"b-Zh*", document_OutputSettings0, true, true, true);
      assertEquals("N}&quot;wJ&quot;b-Zh*", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("&gt;", document_OutputSettings0);
      assertEquals("&amp;gt;", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, "US<ASCII", document_OutputSettings0, true, true, true);
      assertEquals("US<ASCII", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("HenpLwu?k<sHKUL\"|", document_OutputSettings0);
      assertEquals("HenpLwu?k&lt;sHKUL\"|", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape(">", document_OutputSettings0);
      assertEquals("&gt;", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.charset("US-ASCII");
      String string0 = Entities.escape("US-ASCII", document_OutputSettings0);
      assertEquals("US-ASCII", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.charset("838");
      String string0 = Entities.escape("US-ASCII", document_OutputSettings1);
      assertEquals("US-ASCII", string0);
  }
}
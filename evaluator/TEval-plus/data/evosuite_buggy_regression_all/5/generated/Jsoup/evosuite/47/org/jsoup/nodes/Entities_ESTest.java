/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:15:00 GMT 2023
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
      String string0 = Entities.unescape("+[RJmi R9,-Pl6t&O");
      assertEquals("+[RJmi R9,-Pl6t&O", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String string0 = Entities.unescape("&ap;");
      assertEquals("\u2248", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, " tO jbS+(E&E", document_OutputSettings0, true, true, true);
      assertEquals("tO jbS+(E&amp;E", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder("`z<(l|vMriT?{ ");
      Entities.escape(stringBuilder0, "`z<(l|vMriT?{ ", document_OutputSettings0, true, true, false);
      assertEquals("`z<(l|vMriT?{ `z<(l|vMriT?{ ", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder("`z<(l|ri?{ ");
      Entities.escape(stringBuilder0, "`$L?dQXF\"wLhn  nO(", document_OutputSettings0, true, true, true);
      assertEquals("`z<(l|ri?{ `$L?dQXF&quot;wLhn nO(", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder();
      Entities.escape(stringBuilder0, ">P@g_( ytPTd># lp", document_OutputSettings0, true, true, true);
      assertEquals(">P@g_( ytPTd># lp", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("\u00A0", document_OutputSettings0);
      assertEquals("&nbsp;", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Entities.EscapeMode entities_EscapeMode0 = Entities.EscapeMode.xhtml;
      document_OutputSettings0.escapeMode(entities_EscapeMode0);
      String string0 = Entities.escape("\u00A0", document_OutputSettings0);
      assertEquals("&#xa0;", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("hh/Z*<.X", document_OutputSettings0);
      assertEquals("hh/Z*&lt;.X", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape(">P@g_( sTd>#M lp", document_OutputSettings0);
      assertEquals("&gt;P@g_( sTd&gt;#M lp", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      String string0 = Entities.escape("heLhT}nGU^p\"aS", document_OutputSettings0);
      assertEquals("heLhT}nGU^p\"aS", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.charset("US-ASCII");
      String string0 = Entities.escape("US-ASCII", document_OutputSettings0);
      assertEquals("US-ASCII", string0);
  }
}
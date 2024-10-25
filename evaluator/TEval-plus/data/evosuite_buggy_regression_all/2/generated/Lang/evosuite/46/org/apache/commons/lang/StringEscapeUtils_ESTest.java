/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:22:27 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import java.io.Writer;
import org.apache.commons.lang.StringEscapeUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringEscapeUtils_ESTest extends StringEscapeUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJavaScript("w<N7\r8=k[\f^}D");
      assertEquals("w<N7\\r8=k[\\f^}D", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(1093);
      StringEscapeUtils.unescapeJavaScript((Writer) stringWriter0, "Caused by: ");
      assertEquals("Caused by: ", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StringEscapeUtils stringEscapeUtils0 = new StringEscapeUtils();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJavaScript("\u0007oW#5<s6XH='ib");
      assertEquals("\\u0007oW#5<s6XH=\\'ib", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJava("EVQ*8\bDU");
      assertEquals("EVQ*8\\bDU", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.escapeJavaScript((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJava((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringEscapeUtils.escapeJava((Writer) null, "8706");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJavaScript("\t...$");
      assertEquals("\\t...$", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJavaScript("\n");
      assertEquals("\\n", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJava("Q2(\"aexpj");
      assertEquals("Q2(\\\"aexpj", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJava("/WU/`b!l*)");
      assertEquals("\\/WU\\/`b!l*)", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeJava("'x");
      assertEquals("'x", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeJavaScript((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringEscapeUtils.unescapeJava((Writer) null, ")}8jKp5eGz");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.unescapeJava((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = "\\u0,f00";
      // Undeclared exception!
      try { 
        StringEscapeUtils.unescapeJavaScript(string0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unable to parse unicode value: 0,f0
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeJavaScript("\\@000");
      assertEquals("@000", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeJavaScript("EVQ*8\bDU");
      assertEquals("EVQ*8\bDU", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.unescapeJava((Writer) stringWriter0, "$ 5\t");
      assertEquals("$ 5\t", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeJava("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeHtml("3Fz.eo!WXk");
      assertNotNull(string0);
      assertEquals("3Fz.eo!WXk", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeHtml((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringEscapeUtils.escapeHtml((Writer) null, (String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null.
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.escapeHtml((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeHtml("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeHtml((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String string0 = "\\u0";
      String string1 = StringEscapeUtils.escapeCsv(string0);
      // Undeclared exception!
      try { 
        StringEscapeUtils.unescapeHtml((Writer) null, string1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null.
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.unescapeHtml((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.escapeXml((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringEscapeUtils.escapeXml((Writer) null, (String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null.
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.escapeXml((Writer) stringWriter0, "");
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeXml("'x");
      assertEquals("&apos;x", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeXml((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(0);
      StringEscapeUtils.unescapeXml((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringEscapeUtils.unescapeXml((Writer) null, "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null.
         //
         verifyException("org.apache.commons.lang.StringEscapeUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      String string0 = "\\u00";
      StringEscapeUtils.unescapeXml((Writer) stringWriter0, string0);
      assertEquals("\\u00", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeXml("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeXml((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeSql("203");
      assertEquals("203", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeSql((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(659);
      StringEscapeUtils.escapeCsv((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.escapeCsv((Writer) stringWriter0, "");
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      String string0 = StringEscapeUtils.escapeCsv("\"555?u<.");
      assertEquals("\"\"\"555?u<.\"", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeCsv("\",2mM:Vr;\"");
      assertNotNull(string0);
      assertEquals(",2mM:Vr;", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeCsv((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      StringEscapeUtils.unescapeCsv((Writer) stringWriter0, (String) null);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeCsv("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeCsv("QM}B'@5& ");
      assertEquals("QM}B'@5& ", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      String string0 = StringEscapeUtils.unescapeCsv("\"5-5=G?u<.");
      assertEquals("\"5-5=G?u<.", string0);
  }
}

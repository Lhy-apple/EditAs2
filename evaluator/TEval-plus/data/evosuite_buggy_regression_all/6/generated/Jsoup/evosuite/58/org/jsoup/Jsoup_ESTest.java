/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:35:09 GMT 2023
 */

package org.jsoup;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.IllegalCharsetNameException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.jsoup.safety.Whitelist;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Jsoup_ESTest extends Jsoup_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Jsoup.parseBodyFragment("org.jsoup.select.Elements");
      assertFalse(document0.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = Jsoup.parse("", "");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.simpleText();
      String string0 = Jsoup.clean("Fa6XUhgv,O-9 T,}t", whitelist0);
      assertEquals("Fa6XUhgv,O-9 T,}t", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Jsoup.parse("J)mmZ\"8+PlX~UKu'=");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        Jsoup.clean((String) null, "", (Whitelist) null, document_OutputSettings0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String input must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        Jsoup.parse((InputStream) null, "org.jsoup.Jsoup", "org.jsoup.Jsoup", (Parser) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      File file0 = MockFile.createTempFile("W;u]F;$", "W;u]F;$");
      // Undeclared exception!
      try { 
        Jsoup.parse(file0, "rB%dZM", "");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // rB%dZM
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Whitelist whitelist0 = Whitelist.relaxed();
      boolean boolean0 = Jsoup.isValid("", whitelist0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("9!0x{)I5ivF{{}&f", "org.jsoup.Jsoup");
      try { 
        Jsoup.parse((File) mockFile0, "9!0x{)I5ivF{{}&f");
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // File does not exist, and RandomAccessFile is not open in write mode
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockRandomAccessFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      try { 
        Jsoup.parse((URL) null, (-1165));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // URL must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      Document document0 = Jsoup.parse("J)mmZ\"8+PlX~UKu'=", "hr", parser0);
      assertEquals("hr", document0.location());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        Jsoup.connect("org.jsoup.Jsoup");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Malformed URL: org.jsoup.Jsoup
         //
         verifyException("org.jsoup.helper.HttpConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        Jsoup.parse((InputStream) null, "_(5s3{{", "_(5s3{{");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }
}
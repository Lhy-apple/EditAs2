/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:28:05 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.UnsupportedCharsetException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      File file0 = MockFile.createTempFile("audio", "audio");
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      DataInputStream dataInputStream0 = new DataInputStream(mockFileInputStream0);
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) dataInputStream0, "audio", "audio");
        fail("Expecting exception: UnsupportedCharsetException");
      
      } catch(UnsupportedCharsetException e) {
         //
         // audio
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Parser parser0 = Parser.xmlParser();
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "text", "text", parser0);
        fail("Expecting exception: UnsupportedCharsetException");
      
      } catch(UnsupportedCharsetException e) {
         //
         // text
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", (String) null);
      Document document0 = DataUtil.load(file0, (String) null, "UTF-8");
      assertEquals("UTF-8", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((File) null, "Must set charset arg to character set of file to parse. Set to null to attempt to detect from HTML", "Must set charset arg to character set of file to parse. Set to null to attempt to detect from HTML");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      File file0 = MockFile.createTempFile("UTF-8", "UTF-8");
      // Undeclared exception!
      try { 
        DataUtil.load(file0, "UTF-8", "UTF-8");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ByteBuffer byteBuffer0 = ByteBuffer.allocate(49);
      // Undeclared exception!
      try { 
        DataUtil.parseByteData(byteBuffer0, "UTF-8", "UTF-8", (Parser) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.DataUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("*m1S$glaBLL(8");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }
}

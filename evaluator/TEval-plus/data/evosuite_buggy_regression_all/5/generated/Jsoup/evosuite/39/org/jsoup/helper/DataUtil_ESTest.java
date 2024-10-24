/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:14:27 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.nio.ByteBuffer;
import java.nio.charset.IllegalCharsetNameException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-2598), 65275);
      Parser parser0 = Parser.htmlParser();
      // Undeclared exception!
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "", "s", parser0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.ByteArrayInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MockFile mockFile0 = new MockFile("#mBS0}X6#");
      try { 
        DataUtil.load((File) mockFile0, "#mBS0}X6#", "#mBS0}X6#");
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      File file0 = MockFile.createTempFile("cast=", "cast=");
      // Undeclared exception!
      try { 
        DataUtil.load(file0, "cast=", "cast=");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // cast=
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Document document0 = DataUtil.load((InputStream) byteArrayInputStream0, (String) null, "biI]");
      assertEquals("biI]", document0.location());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      File file0 = MockFile.createTempFile("charset", "charset");
      Document document0 = DataUtil.load(file0, "UTF-8", "charset");
      assertEquals("charset", document0.location());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer(pipedInputStream0, (-12));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, 7);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(7, byteBuffer0.capacity());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteBuffer byteBuffer0 = DataUtil.readToByteBuffer(byteArrayInputStream0, (byte)24);
      assertEquals("java.nio.HeapByteBuffer[pos=0 lim=7 cap=7]", byteBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("charset=");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = DataUtil.getCharsetFromContentType("thVSml1vT");
      assertNull(string0);
  }
}

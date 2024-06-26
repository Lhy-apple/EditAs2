/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:59:21 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.jsoup.helper.DataUtil;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DataUtil_ESTest extends DataUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)17, (byte)17);
      // Undeclared exception!
      DataUtil.load((InputStream) byteArrayInputStream0, (String) null, ", state=");
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteBuffer byteBuffer0 = DataUtil.emptyByteBuffer();
      assertEquals(0, byteBuffer0.limit());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.load((File) null, "V:,3%qzywaDJ[XSNf6", (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)17, (byte)17);
      DataUtil.load((InputStream) byteArrayInputStream0, "UTF-8", "UTF-8", parser0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockFile mockFile0 = new MockFile("8OZ7Y:V#", "");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      DataUtil.crossStreams(byteArrayInputStream0, mockPrintStream0);
      assertEquals(9L, mockFile0.length());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      assertFalse(parser0.isTrackErrors());
      assertNotNull(parser0);
      
      DataUtil.load((InputStream) null, (String) null, "V:,3%qzywaDJ[XSNf6", parser0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)16, 1125);
      Parser parser0 = Parser.xmlParser();
      DataUtil.load((InputStream) byteArrayInputStream0, (String) null, "PY", parser0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        DataUtil.readToByteBuffer((InputStream) null, (-30));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DataUtil.getCharsetFromContentType("UTF-8");
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DataUtil.getCharsetFromContentType((String) null);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DataUtil.mimeBoundary();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)17);
      // Undeclared exception!
      try { 
        DataUtil.parseInputStream(byteArrayInputStream0, "", ", state=", parser0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must set charset arg to character set of file to parse. Set to null to attempt to detect from HTML
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Parser parser0 = Parser.xmlParser();
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)16;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        DataUtil.load((InputStream) byteArrayInputStream0, "2!hzA0S\"y", "", parser0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte)17;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)17);
      // Undeclared exception!
      try { 
        DataUtil.parseInputStream(byteArrayInputStream0, "", ", state=", parser0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must set charset arg to character set of file to parse. Set to null to attempt to detect from HTML
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }
}

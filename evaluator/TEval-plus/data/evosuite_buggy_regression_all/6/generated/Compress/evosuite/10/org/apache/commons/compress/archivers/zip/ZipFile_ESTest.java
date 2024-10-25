/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:11:38 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.nio.charset.IllegalCharsetNameException;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.zip.ZipFile;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipFile_ESTest extends ZipFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockFile mockFile0 = new MockFile("@", "@");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(mockFile0, "@");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // @
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      File file0 = MockFile.createTempFile("(s&", "(s&");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(file0);
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("", "");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ZipFile.closeQuietly((ZipFile) null);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile(":?@9=t3}XJ5[B88c[t");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, "&j3R]PKh\"W7sQ{0z");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, "&j3R]PKh\"W7sQ{0z");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(":?@9=t3}XJ5[B88c[t");
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile(":?@9=t3}XJ5[B88c[t");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, "archive's ZIP64 end of central directory locator is corrupt.");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(":?@9=t3}XJ5[B88c[t");
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }
}

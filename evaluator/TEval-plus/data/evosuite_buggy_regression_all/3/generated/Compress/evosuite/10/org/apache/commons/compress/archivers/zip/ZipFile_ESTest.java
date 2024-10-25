/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:27:21 GMT 2023
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
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipLong", "");
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile(mockFile0, " instead of ");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         //  instead of 
         //
         verifyException("java.nio.charset.Charset", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      File file0 = MockFile.createTempFile("failed to skip file name in local file header", "failed to skip file name in local file header");
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
        zipFile0 = new ZipFile("&", "]t'eJdEuCEb7-y@,");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // ]t'eJdEuCEb7-y@,
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
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("&Z-(");
      byte[] byteArray0 = new byte[29];
      FileSystemHandling.appendLineToFile(evoSuiteFile0, "{Co$PKIiLpW");
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("&Z-(");
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
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("&Z-(");
      byte[] byteArray0 = new byte[30];
      byteArray0[1] = (byte)80;
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      ZipFile zipFile0 = null;
      try {
        zipFile0 = new ZipFile("&Z-(");
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // archive is not a ZIP archive
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipFile", e);
      }
  }
}

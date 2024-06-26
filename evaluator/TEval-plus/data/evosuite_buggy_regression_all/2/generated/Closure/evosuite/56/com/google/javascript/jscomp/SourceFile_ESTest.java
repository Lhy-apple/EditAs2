/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:30:09 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.SourceFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SourceFile_ESTest extends SourceFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      SourceFile.Generated sourceFile_Generated0 = new SourceFile.Generated("com.google.javascript.jscomp.SourceFile$Generated", sourceFile_Generator0);
      sourceFile_Generated0.clearCachedSource();
      assertFalse(sourceFile_Generated0.isExtern());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      SourceFile sourceFile0 = SourceFile.fromFile("r", charset0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("Cew2");
      sourceFile0.clearCachedSource();
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        SourceFile.fromGenerator((String) null, (SourceFile.Generator) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StringReader stringReader0 = new StringReader("s]ua");
      SourceFile sourceFile0 = SourceFile.fromReader("s]ua", stringReader0);
      Region region0 = sourceFile0.getRegion((-15));
      assertEquals("s]ua", region0.getSourceExcerpt());
      assertEquals(1, region0.getEndingLineNumber());
      assertNotNull(region0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("a source must have a name", "a source must have a name", "a source must have a name");
      sourceFile_Preloaded0.clearCachedSource();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("B}^++aXHW", "com.google.javascript.jscomp.SourceFile");
      String string0 = sourceFile_Preloaded0.getCodeNoCache();
      assertEquals("com.google.javascript.jscomp.SourceFile", string0);
      assertEquals("B}^++aXHW", sourceFile_Preloaded0.getOriginalPath());
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("com.google.javascript.jscomp.SourceFile$Generated", "com.google.javascript.jscomp.SourceFile$Generated");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(mockFile0);
      sourceFile_OnDisk0.getName();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = (SourceFile.Preloaded)SourceFile.fromCode("nBY;CNXj Ivy{{~K", "nBY;CNXj Ivy{{~K", "nBY;CNXj Ivy{{~K");
      assertFalse(sourceFile_Preloaded0.isExtern());
      
      sourceFile_Preloaded0.setIsExtern(true);
      assertTrue(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("a source must have a name", "a source must have a name", "a source must have a name");
      sourceFile_Preloaded0.toString();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        SourceFile.fromInputStream("\n", "\n", (InputStream) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.Reader", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        SourceFile.fromInputStream("M5y0r `*:21{K+pd2wR", (InputStream) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.Reader", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFile mockFile0 = new MockFile("com.google.javascript.jscomp.SourceFile$Generated", "com.google.javascript.jscomp.SourceFile$Generated");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(mockFile0);
      boolean boolean0 = sourceFile_OnDisk0.isExtern();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SourceFile sourceFile0 = null;
      try {
        sourceFile0 = new SourceFile("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("a source must have a 0ame", "a source must have a 0ame", "a source must have a 0ame");
      sourceFile_Preloaded0.getNumLines();
      // Undeclared exception!
      try { 
        sourceFile_Preloaded0.getLineOffset((-1689));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected line number between 1 and 1
         // Actual: -1689
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromCode("BT;CNXW Ivy{`", "BT;CNXW Ivy{`", "BT;CNXW Ivy{`");
      sourceFile0.getLineOffset(1);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("'\n", "'\n", "'\n");
      sourceFile_Preloaded0.getNumLines();
      int int0 = sourceFile_Preloaded0.getNumLines();
      assertFalse(sourceFile_Preloaded0.isExtern());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\n", "\n", "\nActual: ");
      // Undeclared exception!
      try { 
        sourceFile_Preloaded0.getLineOffset(28);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected line number between 1 and 2
         // Actual: 28
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      File file0 = MockFile.createTempFile("a source must have a name", (String) null);
      SourceFile sourceFile0 = SourceFile.fromFile(file0);
      String string0 = sourceFile0.getOriginalPath();
      assertNotNull(string0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("a source must have a name", "a source must have a name", "a source must have a name");
      sourceFile_Preloaded0.getOriginalPath();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      MockFile mockFile0 = new MockFile("+\n");
      SourceFile sourceFile0 = SourceFile.fromFile((File) mockFile0);
      try { 
        sourceFile0.getCodeReader();
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("B}^++aXHW", "com.google.javascript.jscomp.SourceFile");
      boolean boolean0 = sourceFile_Preloaded0.hasSourceInMemory();
      assertTrue(boolean0);
      assertEquals("B}^++aXHW", sourceFile_Preloaded0.getOriginalPath());
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromCode("\n", "\n", "\n");
      String string0 = sourceFile0.getLine((-16));
      assertFalse(sourceFile0.isExtern());
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromCode("\n", "\n", "\n");
      String string0 = sourceFile0.getLine(3481);
      assertNull(string0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("a source must have a name", "a source must have a name", "a source must have a name");
      sourceFile_Preloaded0.getLine((-1));
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("'\n", "'\n", "'\n");
      Region region0 = sourceFile_Preloaded0.getRegion((-752));
      assertNotNull(region0);
      assertEquals("'", region0.getSourceExcerpt());
      assertEquals(2, region0.getEndingLineNumber());
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\n", "\n", "\n");
      sourceFile_Preloaded0.getRegion(818);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn(":").when(sourceFile_Generator0).getCode();
      SourceFile.Generated sourceFile_Generated0 = new SourceFile.Generated(":", sourceFile_Generator0);
      sourceFile_Generated0.getCode();
      String string0 = sourceFile_Generated0.getCode();
      assertNotNull(string0);
      assertFalse(sourceFile_Generated0.isExtern());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MockFile mockFile0 = new MockFile("start index");
      File file0 = MockFile.createTempFile("start index", "start index", (File) mockFile0);
      SourceFile sourceFile0 = SourceFile.fromFile(file0, (Charset) null);
      sourceFile0.getRegion(3858);
      Region region0 = sourceFile0.getRegion(3858);
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      File file0 = MockFile.createTempFile("X~`}", "X~`}");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      String string0 = sourceFile_OnDisk0.getCode();
      assertEquals("", string0);
      assertNotNull(string0);
      
      sourceFile_OnDisk0.getCodeReader();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }
}

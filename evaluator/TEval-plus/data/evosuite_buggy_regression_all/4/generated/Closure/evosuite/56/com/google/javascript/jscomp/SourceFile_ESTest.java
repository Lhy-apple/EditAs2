/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:13:41 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.SourceFile;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SourceFile_ESTest extends SourceFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      SourceFile.Generated sourceFile_Generated0 = new SourceFile.Generated("\nActual: ", sourceFile_Generator0);
      sourceFile_Generated0.clearCachedSource();
      assertFalse(sourceFile_Generated0.isExtern());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      File file0 = MockFile.createTempFile("uv]d0", "");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      sourceFile_OnDisk0.getRegion(16);
      Reader reader0 = sourceFile_OnDisk0.getCodeReader();
      assertNotNull(reader0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      SourceFile sourceFile0 = SourceFile.fromFile(">,*.F8'c89,Eyr2", charset0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      File file0 = MockFile.createTempFile("uv]d0", "");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(file0);
      sourceFile_OnDisk0.clearCachedSource();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromCode("com.google.common.io.ByteStreams", "com.google.common.io.ByteStreams", "com.google.common.io.ByteStreams");
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StringReader stringReader0 = new StringReader("\nActual:j");
      SourceFile sourceFile0 = SourceFile.fromReader("\nActual:j", stringReader0);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActual: ", "\nActual: ", "\nActual: ");
      sourceFile_Preloaded0.getCodeReader();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nAcual ", "\nAcual ", "\nAcual ");
      sourceFile_Preloaded0.clearCachedSource();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("}v", "}v");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(mockFile0);
      sourceFile_OnDisk0.getCodeNoCache();
      assertFalse(sourceFile_OnDisk0.isExtern());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("Expected line number between 1 and ", "={VG");
      SourceFile sourceFile0 = SourceFile.fromFile((File) mockFile0);
      sourceFile0.getName();
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nAcual ", "\nAcual ", "\nAcual ");
      assertFalse(sourceFile_Preloaded0.isExtern());
      
      sourceFile_Preloaded0.setIsExtern(true);
      assertTrue(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("+i9`dtLRk%v!`A*Vgr", "+i9`dtLRk%v!`A*Vgr");
      sourceFile_Preloaded0.toString();
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      FileDescriptor fileDescriptor0 = new FileDescriptor();
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(fileDescriptor0);
      try { 
        SourceFile.fromInputStream("a source must have a name", "a source must have a name", (InputStream) mockFileInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.NativeMockedIO", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("\nc'&tual:");
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      File file0 = MockFile.createTempFile("\nAcual ", "\nAcual ");
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      SourceFile sourceFile0 = SourceFile.fromInputStream("\nAcual ", (InputStream) mockFileInputStream0);
      assertEquals("\nAcual ", sourceFile0.getOriginalPath());
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("com.google.javascript.jscomp.SourceFile$Preloaded", "com.google.javascript.jscomp.SourceFile$Preloaded");
      boolean boolean0 = sourceFile_Preloaded0.isExtern();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SourceFile sourceFile0 = null;
      try {
        sourceFile0 = new SourceFile((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      try { 
        SourceFile.fromCode("", "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActual: ", "\nActual: ");
      int int0 = sourceFile_Preloaded0.getNumLines();
      assertEquals(2, int0);
      
      int int1 = sourceFile_Preloaded0.getLineOffset(1);
      assertEquals(0, int1);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("com.google.javascript.jscomp.SourceFile$Generated", "com.google.javascript.jscomp.SourceFile$Generated");
      sourceFile_Preloaded0.getLineOffset(1);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nAcQtual: ", "\nAcQtual: ", "\nAcQtual: ");
      // Undeclared exception!
      try { 
        sourceFile_Preloaded0.getLineOffset((-511));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected line number between 1 and 2
         // Actual: -511
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActual: ", "\nActual: ", "\nActual: ");
      // Undeclared exception!
      try { 
        sourceFile_Preloaded0.getLineOffset(3086);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected line number between 1 and 2
         // Actual: 3086
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActual: ", "\nActual: ");
      sourceFile_Preloaded0.getNumLines();
      int int0 = sourceFile_Preloaded0.getNumLines();
      assertEquals(2, int0);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromGenerator("$BB[:hMpfO", (SourceFile.Generator) null);
      String string0 = sourceFile0.getOriginalPath();
      assertFalse(sourceFile0.isExtern());
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromCode("KMu'Vf~5WpV'8$!I@AO", "h:");
      String string0 = sourceFile0.getOriginalPath();
      assertFalse(sourceFile0.isExtern());
      assertEquals("KMu'Vf~5WpV'8$!I@AO", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActdual: ", "\nActdual: ");
      String string0 = sourceFile_Preloaded0.getLine((-479));
      assertNotNull(string0);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActual: ", "\nActual: ");
      String string0 = sourceFile_Preloaded0.getLine(1362);
      assertFalse(sourceFile_Preloaded0.isExtern());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("+i9`dtLRk%v!`A*Vgr", "+i9`dtLRk%v!`A*Vgr");
      sourceFile_Preloaded0.getLine((-1242));
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActual: ", "\nActual: ");
      Region region0 = sourceFile_Preloaded0.getRegion((-1589));
      assertNotNull(region0);
      assertEquals("\nActual: ", region0.getSourceExcerpt());
      assertEquals(2, region0.getEndingLineNumber());
      assertFalse(sourceFile_Preloaded0.isExtern());
      assertEquals(1, region0.getBeginningLineNumber());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\nActdual: ", "\nActdual: ");
      Region region0 = sourceFile_Preloaded0.getRegion(1996);
      assertNull(region0);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SourceFile.Preloaded sourceFile_Preloaded0 = new SourceFile.Preloaded("\n", "\n", "\n");
      Region region0 = sourceFile_Preloaded0.getRegion((-3064));
      assertEquals(2, region0.getEndingLineNumber());
      assertNotNull(region0);
      assertFalse(sourceFile_Preloaded0.isExtern());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      doReturn("VWgln6a@K0$EvC").when(sourceFile_Generator0).getCode();
      SourceFile sourceFile0 = SourceFile.fromGenerator("-/WtS&o-|^w", sourceFile_Generator0);
      sourceFile0.getLine(3);
      Region region0 = sourceFile0.getRegion(10);
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SourceFile sourceFile0 = SourceFile.fromFile("b~7x%P~i\"7@PYv6fYRtb", (Charset) null);
      assertFalse(sourceFile0.isExtern());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      MockFile mockFile0 = new MockFile("Cv");
      SourceFile.OnDisk sourceFile_OnDisk0 = new SourceFile.OnDisk(mockFile0);
      try { 
        sourceFile_OnDisk0.getCodeReader();
        fail("Expecting exception: FileNotFoundException");
      
      } catch(FileNotFoundException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:47:51 GMT 2023
 */

package com.fasterxml.jackson.core.sym;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ByteQuadsCanonicalizer_ESTest extends ByteQuadsCanonicalizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(180);
      int[] intArray0 = new int[25];
      byteQuadsCanonicalizer1.addName("", intArray0, 7);
      byteQuadsCanonicalizer1.findName(intArray0, 7);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(571);
      int int0 = byteQuadsCanonicalizer0.hashSeed();
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(571, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(1532);
      int int0 = byteQuadsCanonicalizer0.bucketCount();
      assertEquals(1532, byteQuadsCanonicalizer0.hashSeed());
      assertEquals(0, int0);
      assertEquals(0, byteQuadsCanonicalizer0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(77);
      byteQuadsCanonicalizer1.addName("W<_w?=?s -E|=_", 77);
      byteQuadsCanonicalizer1.release();
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0.release();
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(88);
      byteQuadsCanonicalizer1.release();
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(0);
      int int0 = byteQuadsCanonicalizer1.size();
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, int0);
      assertFalse(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(6);
      int int0 = byteQuadsCanonicalizer0.size();
      assertEquals(0, int0);
      assertEquals(6, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1836);
      byteQuadsCanonicalizer1.addName("+C", 1836, 0);
      byteQuadsCanonicalizer1.addName("+C", 1836);
      byteQuadsCanonicalizer1.toString();
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[8];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.toString();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(213);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(213);
      byteQuadsCanonicalizer1.findName(213);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(213, byteQuadsCanonicalizer1.hashSeed());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(1);
      int[] intArray0 = new int[4];
      intArray0[3] = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      assertEquals(1, byteQuadsCanonicalizer0.hashSeed());
      
      int[] intArray1 = new int[11];
      intArray1[0] = 1;
      byteQuadsCanonicalizer0.findName(intArray1, 1);
      assertEquals(0, byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(77);
      byteQuadsCanonicalizer1.addName("$VALUES", 77);
      byteQuadsCanonicalizer1.findName(77);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(213);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(213);
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", 213, 0);
      byteQuadsCanonicalizer1.findName(213);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1420));
      byteQuadsCanonicalizer1.addName("U", (-1420), 0);
      byteQuadsCanonicalizer1.addName("U", (-1420));
      byteQuadsCanonicalizer1.findName((-1420));
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(213);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(213);
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", 213, 0);
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", 213, 467);
      byteQuadsCanonicalizer1.findName(213);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(3);
      int[] intArray0 = new int[15];
      byteQuadsCanonicalizer0._hashSize = 3;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(3, 1);
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(3, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[21];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(3);
      int[] intArray0 = new int[15];
      intArray0[3] = 3;
      byteQuadsCanonicalizer0._hashSize = 3;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(3, 1);
      assertEquals((-6), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(153);
      byteQuadsCanonicalizer1.addName("vntxg4oss|", 153, 1305);
      byteQuadsCanonicalizer1.findName(33, 1305);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(3);
      int[] intArray0 = new int[15];
      intArray0[3] = 3;
      byteQuadsCanonicalizer0._hashSize = 3;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(1, 0, 3);
      assertEquals((-6), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(88);
      byteQuadsCanonicalizer1.addName("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", 88, 88, 88);
      byteQuadsCanonicalizer1.findName(88, 22, 560);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(3);
      int[] intArray0 = new int[20];
      intArray0[3] = 3;
      byteQuadsCanonicalizer0._hashSize = 3;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 3);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(100);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(100);
      byteQuadsCanonicalizer1.findName(100, 100, 100);
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(100, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[39];
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(byteQuadsCanonicalizer0._hashArea, 5);
      assertEquals(2, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(16);
      int[] intArray0 = new int[23];
      byteQuadsCanonicalizer1.addName("'Er8", intArray0, 4);
      byteQuadsCanonicalizer1.findName(intArray0, 16);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(16);
      int[] intArray0 = new int[23];
      byteQuadsCanonicalizer1.addName("'Er8", intArray0, 4);
      byteQuadsCanonicalizer1.addName("'Er8", intArray0, 16);
      byteQuadsCanonicalizer1.findName(intArray0, 16);
      assertEquals(1, byteQuadsCanonicalizer1.secondaryCount());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(213);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(213);
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", 213, 0);
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", 213, 467);
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", 213);
      byteQuadsCanonicalizer1.findName(213);
      assertEquals(3, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(9);
      int[] intArray0 = new int[9];
      intArray0[3] = 9;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0._spilloverEnd = 9;
      byteQuadsCanonicalizer0.findName(9);
      assertEquals(2, byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2383);
      int[] intArray0 = new int[4];
      intArray0[0] = 2383;
      intArray0[3] = 2383;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0._spilloverEnd = 2383;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(2383);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 4
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      byteQuadsCanonicalizer0._tertiaryStart = 2;
      int[] intArray0 = new int[21];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      int[] intArray1 = new int[6];
      intArray1[1] = 2;
      String string0 = byteQuadsCanonicalizer0.findName(intArray1, 2);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[21];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._spilloverEnd = 37;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(2, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 22
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[21];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      intArray0[15] = 2;
      int[] intArray1 = new int[6];
      intArray1[1] = 2;
      byteQuadsCanonicalizer0._spilloverEnd = 1183;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray1, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 22
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2383);
      int[] intArray0 = new int[9];
      intArray0[0] = 2383;
      intArray0[3] = 2383;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      String string0 = byteQuadsCanonicalizer0.findName(intArray0, 3);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      byteQuadsCanonicalizer0._tertiaryStart = 2;
      int[] intArray0 = new int[11];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(0, (-441), 32768);
      assertEquals(2, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[21];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      int[] intArray1 = new int[6];
      intArray1[1] = 2;
      byteQuadsCanonicalizer0._spilloverEnd = 1183;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray1, 3);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 22
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2383);
      int[] intArray0 = new int[9];
      intArray0[0] = 2383;
      intArray0[3] = 2383;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0._spilloverEnd = 2383;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 3);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 12
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(9);
      int[] intArray0 = new int[13];
      intArray0[3] = 9;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      int[] intArray1 = new int[11];
      intArray1[0] = 9;
      String string0 = byteQuadsCanonicalizer0.findName(intArray1, 8);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(9);
      int[] intArray0 = new int[6];
      intArray0[3] = 9;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      int[] intArray1 = new int[11];
      byteQuadsCanonicalizer0._spilloverEnd = 9;
      intArray1[0] = 9;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray1, 9);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 7
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      byteQuadsCanonicalizer0._spilloverEnd = 1345;
      int[] intArray0 = new int[39];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      int[] intArray1 = new int[6];
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray1, 5);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 42
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(5);
      int[] intArray0 = new int[9];
      intArray0[3] = 5;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      int[] intArray1 = new int[11];
      byteQuadsCanonicalizer0._spilloverEnd = 5;
      intArray1[0] = 5;
      String string0 = byteQuadsCanonicalizer0.findName(intArray1, 5);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(16);
      int[] intArray0 = new int[23];
      byteQuadsCanonicalizer1.addName("'Er8", intArray0, 4);
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(6);
      int[] intArray0 = new int[13];
      byteQuadsCanonicalizer1.addName("7eG5T^.#5^wp", intArray0, 6);
      byteQuadsCanonicalizer1.findName(intArray0, 6);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(64);
      byteQuadsCanonicalizer1._count = 64;
      byteQuadsCanonicalizer1.addName("lG", 65);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("lG", 64);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=65, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(64);
      byteQuadsCanonicalizer1._count = 64;
      byteQuadsCanonicalizer1.addName("lG", 65, 64, 64);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("lG", 64);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=65, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(77);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(77);
      byteQuadsCanonicalizer1._hashSize = (-817);
      byteQuadsCanonicalizer1.addName("$VLUES", 77);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(35);
      byteQuadsCanonicalizer1._count = 35;
      byteQuadsCanonicalizer1.addName("", 35, 0);
      assertEquals(36, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2383);
      int[] intArray0 = new int[7];
      intArray0[3] = 2383;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("d", 2383);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      int[] intArray0 = new int[8];
      intArray0[3] = 2;
      byteQuadsCanonicalizer0._hashSize = 2;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("1D/ iU3D}+", 2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[2];
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("(a~hkav':hFOwv'(U [", intArray0, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(76);
      byteQuadsCanonicalizer1._count = 76;
      byteQuadsCanonicalizer1.addName("$VUE'", 76, 0);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("$VUE'", 76);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=77, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(77);
      int[] intArray0 = new int[10];
      String[] stringArray0 = new String[8];
      byteQuadsCanonicalizer0._names = stringArray0;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0._spilloverEnd = 3;
      byteQuadsCanonicalizer0.addName("hg", 77);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName(stringArray0[1], 0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 1667;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0._reportTooManyCollisions();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Spill-over slots in symbol table with 0 entries, hash area of 1667 slots is now full (all 208 slots -- suspect a DoS attack based on hash collisions. You can disable the check via `JsonFactory.Feature.FAIL_ON_SYMBOL_HASH_OVERFLOW`
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(2175);
      assertEquals(6, int0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(515);
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(4667);
      assertEquals(7, int0);
  }
}
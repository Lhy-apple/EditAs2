/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:37:00 GMT 2023
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
      int int0 = byteQuadsCanonicalizer0.hashSeed();
      assertEquals(839877741, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int int0 = byteQuadsCanonicalizer0.bucketCount();
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName(",", 0, 6000, 6000);
      byteQuadsCanonicalizer1.release();
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0.release();
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1289);
      byteQuadsCanonicalizer1.release();
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertFalse(byteQuadsCanonicalizer1.maybeDirty());
      assertEquals(0, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1289);
      byteQuadsCanonicalizer1._count = 6000;
      byteQuadsCanonicalizer1.addName("CANONICALIZE_FIELD_NAMES", 6000, 6000, 64);
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
      
      byteQuadsCanonicalizer1.release();
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(51);
      int int0 = byteQuadsCanonicalizer1.size();
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertFalse(byteQuadsCanonicalizer1.maybeDirty());
      assertEquals(0, int0);
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int int0 = byteQuadsCanonicalizer0.size();
      assertEquals(0, int0);
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("C", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("C", 6000, 16, 7);
      byteQuadsCanonicalizer1.toString();
      assertEquals(7, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("C", 0, 6000, 6000);
      byteQuadsCanonicalizer1.toString();
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.findName(intArray0, 0);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1);
      byteQuadsCanonicalizer1._hashSize = 1;
      int[] intArray0 = new int[17];
      byteQuadsCanonicalizer1.addName("`8.Jc?j@A3rl)>utjcxW", 1);
      byteQuadsCanonicalizer1.findName(intArray0, 1);
      assertEquals(1, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-634));
      byteQuadsCanonicalizer1.addName("CANONICALIZE_FIELD_NAMES", (-634));
      byteQuadsCanonicalizer1.findName((-634));
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.findName(intArray0, 0);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1289);
      byteQuadsCanonicalizer1.addName("CANONICALIZE_FIELD_NAMES", 6000, 1289);
      byteQuadsCanonicalizer1.addName("CANONICALIZE_FIELD_NAMES", 1289);
      byteQuadsCanonicalizer1.findName(1289);
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-886));
      int[] intArray0 = new int[17];
      intArray0[0] = (-886);
      intArray0[3] = (-886);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      String string0 = byteQuadsCanonicalizer0.findName(intArray0, (-886));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.findName(intArray0, 2);
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(2);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(2);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.addName(" entries, hash area of ", intArray0, 2);
      byteQuadsCanonicalizer1.findName(intArray0, 2);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=7, hashSize=7, 0/0/0/99 pri/sec/ter/spill (=0), total:99]", 7, 7);
      byteQuadsCanonicalizer1.findName(7, 180);
      assertEquals(99, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName(",", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName(",", 3022, (-967));
      byteQuadsCanonicalizer1.findName(6000, (-967));
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("1r.a)c%w[>RlRmS?(xZ", 16);
      byteQuadsCanonicalizer1.addName("1r.a)c%w[>RlRmS?(xZ", 2, 16);
      byteQuadsCanonicalizer1.findName(2, 1);
      assertEquals(2, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 6000, 16, 7);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.findName(intArray0, 2);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer1.findName(intArray0, 3);
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(3);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(3);
      int[] intArray0 = new int[15];
      byteQuadsCanonicalizer1.addName("7Ba1iNo&RV])", intArray0, 3);
      byteQuadsCanonicalizer1.findName(intArray0, 3);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(20);
      String[] stringArray0 = new String[15];
      byteQuadsCanonicalizer1._hashSize = 20;
      byteQuadsCanonicalizer1.addName(stringArray0[12], 20, 16, 20);
      byteQuadsCanonicalizer1.findName(20, 16, 16);
      assertEquals(20, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("USE_THREAD_LOCAL_FOR_BUFFER_RECYCLING", 6000);
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer1.findName(intArray0, 3);
      assertEquals(1, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("fi9'le", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("fi9'le", 6000, 16, 7);
      byteQuadsCanonicalizer1.findName(6000, 16, 16);
      assertEquals(7, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("C", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("C", 6000, 16, 7);
      byteQuadsCanonicalizer1.findName(6000, 2984, 2984);
      assertEquals(7, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("!", 0, 4, 4);
      byteQuadsCanonicalizer1.addName("!", 7, 7, 640);
      byteQuadsCanonicalizer1.findName(7, 7, 640);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      int[] intArray0 = new int[13];
      byteQuadsCanonicalizer1.findName(intArray0, 7);
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(10);
      int[] intArray0 = new int[23];
      byteQuadsCanonicalizer1.addName("AQuB", intArray0, 10);
      assertEquals(1, byteQuadsCanonicalizer1.size());
      
      byteQuadsCanonicalizer1.addName("AQuB", intArray0, 17);
      byteQuadsCanonicalizer1.findName(intArray0, 17);
      assertEquals(0, byteQuadsCanonicalizer1.tertiaryCount());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(10);
      int[] intArray0 = new int[23];
      byteQuadsCanonicalizer1.addName(")A]fS=Z4ygsDF=", intArray0, 12);
      byteQuadsCanonicalizer1.addName("", intArray0, 12);
      byteQuadsCanonicalizer1.findName(intArray0, 10);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 16);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.findName(intArray0, 0);
      assertEquals(3, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("C", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("C", 6000, 16, 7);
      byteQuadsCanonicalizer1.addName("C", 6000);
      byteQuadsCanonicalizer1.findName(6000);
      assertEquals(99, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-886));
      int[] intArray0 = new int[17];
      byteQuadsCanonicalizer0._spilloverEnd = 2305;
      intArray0[0] = (-886);
      intArray0[3] = (-886);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, (-886));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 20
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 6000, 16, 7);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", intArray0, 7);
      byteQuadsCanonicalizer1.findName(intArray0, 2);
      assertEquals(7, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("fi9'le", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("fi9'le", 6000, 16, 7);
      byteQuadsCanonicalizer1.addName("fi9'le", 16, 7);
      byteQuadsCanonicalizer1.findName(16, 7);
      assertEquals(3, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 6000, 16, 7);
      byteQuadsCanonicalizer1.addName("hf%D7Vb#?qy)t", 16);
      byteQuadsCanonicalizer1.findName(2873, 16);
      assertEquals(4, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("A", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("A", 6000, 16, 7);
      byteQuadsCanonicalizer1.addName("A", 16);
      byteQuadsCanonicalizer1.findName(0, 6000);
      assertEquals(7, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("USE_THREAD_LOCAL_FOR_BUFFER_RECYCLING", 0, 6000, 6000);
      byteQuadsCanonicalizer1.addName("USE_THREAD_LOCAL_FOR_BUFFER_RECYCLING", 6000);
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer1.findName(intArray0, 3);
      assertEquals(7, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-871));
      int[] intArray0 = new int[17];
      byteQuadsCanonicalizer0._spilloverEnd = 2305;
      intArray0[0] = (-871);
      intArray0[3] = (-871);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 3);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 20
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("A", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("A", 6000, 16, 7);
      byteQuadsCanonicalizer1.addName("A", 16);
      byteQuadsCanonicalizer1.findName(0, 329, 16);
      assertEquals(99, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._tertiaryStart = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 6000, 16, 7);
      int[] intArray0 = new int[9];
      byteQuadsCanonicalizer1.addName("hg%kD7Vu6.y)t", 6000);
      byteQuadsCanonicalizer1.findName(intArray0, 7);
      assertEquals(4, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("!", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("!", 6000, 7, 7);
      int[] intArray0 = new int[13];
      intArray0[0] = 7;
      intArray0[1] = 6000;
      intArray0[2] = 7;
      intArray0[3] = 7;
      byteQuadsCanonicalizer1.addName("!", intArray0, 7);
      byteQuadsCanonicalizer1.findName(intArray0, 7);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("!", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("!", 6000, 7, 7);
      int[] intArray0 = new int[13];
      intArray0[0] = 7;
      intArray0[1] = 6000;
      intArray0[2] = 7;
      intArray0[3] = 7;
      byteQuadsCanonicalizer1.addName("!", intArray0, 7);
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(1831);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(4);
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer1.addName("x`i<$=O<cR5M,WC", intArray0, 4);
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(5);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(5);
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer1.addName(" entries, hash area of ", intArray0, 5);
      byteQuadsCanonicalizer1.findName(intArray0, 5);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(6);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(6);
      int[] intArray0 = new int[10];
      byteQuadsCanonicalizer1.addName(" entries, hash area of ", intArray0, 6);
      byteQuadsCanonicalizer1.findName(intArray0, 6);
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(26);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(26);
      byteQuadsCanonicalizer1.addName("iyQfNGHTyp", 26, 26);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(6);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName(", copyCount=", 6, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(45);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(45);
      byteQuadsCanonicalizer1._count = 45;
      byteQuadsCanonicalizer1.addName("G<+(]JP", 45, 45);
      assertEquals(46, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1259));
      int[] intArray0 = new int[5];
      intArray0[3] = (-1259);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName(" slots -- suspect a DoS attack based on hash collisions.", (-1259));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1259));
      int[] intArray0 = new int[5];
      byteQuadsCanonicalizer0._spilloverEnd = (-1259);
      intArray0[3] = (-1259);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName(" slots -- suspect a DoS attack based on hash collisions.", (-1259));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1259
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[4];
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("[%s: size=%d, hashSize=%d, %d/%d/%d/%d pri/sec/ter/spill (=%s), total:%d]", intArray0, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1._count = 7;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=7, hashSize=7, 0/0/0/99 pri/sec/ter/spill (=0), total:99]", 7, 7);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName((String) null, 439, 407);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=8, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1289);
      byteQuadsCanonicalizer1._count = 6000;
      byteQuadsCanonicalizer1.addName("CANONICALIZE_FIELD_NAMES", 6000, 6000, 64);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("CANONICALIZE_FIELD_NAMES", 1289);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=6001, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(7);
      byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 7, 7, 2873);
      byteQuadsCanonicalizer1._hashSize = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 0, 6000, 6000);
      byteQuadsCanonicalizer1._secondaryStart = 7;
      byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 6000, 16, 7);
      byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 16);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("hg%kD7Vb+#.y)t", 3006);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(6);
      int[] intArray0 = new int[6];
      byteQuadsCanonicalizer1._count = 422;
      byteQuadsCanonicalizer1.addName("", intArray0, 6);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("", 6, 6);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=423, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 2445;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0._reportTooManyCollisions();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Spill-over slots in symbol table with 0 entries, hash area of 2445 slots is now full (all 305 slots -- suspect a DoS attack based on hash collisions. You can disable the check via `JsonFactory.Feature.FAIL_ON_SYMBOL_HASH_OVERFLOW`
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(65599);
      assertEquals(7, int0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(515);
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(2648);
      assertEquals(6, int0);
  }
}
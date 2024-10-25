/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:19:39 GMT 2023
 */

package com.fasterxml.jackson.core.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.TextBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TextBuffer_ESTest extends TextBuffer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.toString();
      String string0 = textBuffer0.contentsAsString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      try { 
        textBuffer0.contentsAsDouble();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.append(charArray0, 1, 3196);
      String string0 = textBuffer0.toString();
      textBuffer0.append(string0, 3, 3);
      textBuffer0.resetWithString(string0);
      assertFalse(textBuffer0.hasTextAsCharacters());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.getCurrentSegmentSize();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.releaseBuffers();
      assertTrue(textBuffer0.hasTextAsCharacters());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(200, charArray0.length);
      
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      textBuffer0.resetWithEmpty();
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 1, 3);
      textBuffer0.toString();
      assertEquals(1, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithCopy(charArray0, 3, 3);
      assertEquals(3, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      textBuffer0.resetWithCopy(charArray0, 3, 0);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.contentsAsArray();
      // Undeclared exception!
      try { 
        textBuffer0.resetWithCopy(charArray0, 1, 199);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithString("2.2250738585072012e-308");
      assertEquals(23, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(1000, charArray0.length);
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler(57, 2076);
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.size();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.contentsAsString();
      assertFalse(textBuffer0.hasTextAsCharacters());
      
      textBuffer0.getTextBuffer();
      int int0 = textBuffer0.size();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsString();
      textBuffer0.getCurrentSegment();
      textBuffer0.size();
      assertFalse(textBuffer0.hasTextAsCharacters());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(200, charArray0.length);
      
      int int0 = textBuffer0.getTextOffset();
      assertEquals(0, int0);
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler(2050, 262145);
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.getTextOffset();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('u');
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('u');
      textBuffer0.contentsAsArray();
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.toString();
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getTextBuffer();
      assertNull(charArray0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.getTextBuffer();
      textBuffer0.getTextBuffer();
      assertEquals(200, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = textBuffer0.getTextBuffer();
      assertEquals(200, charArray0.length);
      assertNotNull(charArray0);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.getTextBuffer();
      textBuffer0.contentsAsString();
      assertEquals(200, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsArray();
      char[] charArray0 = textBuffer0.contentsAsArray();
      assertEquals(0, charArray0.length);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsArray();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 1, 0);
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(200, charArray0.length);
      
      textBuffer0.ensureNotShared();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append(charArray0, 1, 3129);
      textBuffer0.append('H');
      assertEquals(3130, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.append(charArray0, 1383, 3);
      textBuffer0.append(charArray0, 1, 3086);
      assertEquals(3089, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.append(charArray0, 1, 3189);
      textBuffer0.append(charArray0, 2, 3189);
      assertEquals(3189, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append("SFC(H", 3, 1);
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.append(charArray0, 1, 3157);
      String string0 = textBuffer0.toString();
      TextBuffer textBuffer1 = new TextBuffer(bufferRecycler0);
      textBuffer1.emptyAndGetCurrentSegment();
      // Undeclared exception!
      try { 
        textBuffer1.append(string0, 0, 276426);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      char[] charArray1 = textBuffer0.getCurrentSegment();
      assertSame(charArray1, charArray0);
      assertEquals(0, textBuffer0.size());
      assertEquals(200, charArray1.length);
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithEmpty();
      assertEquals(0, textBuffer0.getTextOffset());
      
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(200, charArray0.length);
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.append(charArray0, 3, 917);
      textBuffer0.getCurrentSegment();
      assertEquals(917, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(1000, charArray0.length);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.expandCurrentSegment(262144);
      textBuffer0.finishCurrentSegment();
      assertEquals(262144, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      assertEquals(46198, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = textBuffer0.expandCurrentSegment(1);
      assertEquals(0, textBuffer0.getTextOffset());
      assertEquals(200, charArray0.length);
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 0, 999);
      // Undeclared exception!
      try { 
        textBuffer0.append('\\');
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('u');
      char[] charArray0 = textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 'u', 3);
      textBuffer0.append('u');
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.append(charArray0, 1, 3196);
      String string0 = textBuffer0.toString();
      textBuffer0.append(string0, 3, 3);
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.setCurrentLength(262145);
      textBuffer0.append(string0, 3, 1);
      String string1 = textBuffer0.contentsAsString();
      textBuffer0.append(string1, 2, 262145);
      assertEquals(619531, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.toString();
      char[] charArray0 = textBuffer0.contentsAsArray();
      assertArrayEquals(new char[] {}, charArray0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(0);
      textBuffer0.resetWithShared(charArray0, 0, 36680);
      textBuffer0.contentsAsArray();
      assertEquals(36680, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 3, 11389);
      textBuffer0.contentsAsArray();
      assertEquals(11389, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      char[] charArray0 = textBuffer0.contentsAsArray();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
      assertNotNull(charArray0);
  }
}

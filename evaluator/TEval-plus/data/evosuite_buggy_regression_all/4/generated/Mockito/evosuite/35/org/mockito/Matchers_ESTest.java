/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:17:46 GMT 2023
 */

package org.mockito;

import org.junit.Test;
import static org.junit.Assert.*;
import java.lang.reflect.Array;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.hamcrest.Matcher;
import org.hamcrest.SelfDescribing;
import org.hamcrest.beans.HasPropertyWithValue;
import org.hamcrest.collection.IsArrayContaining;
import org.hamcrest.core.AllOf;
import org.hamcrest.core.AnyOf;
import org.hamcrest.core.Is;
import org.hamcrest.core.IsAnything;
import org.hamcrest.core.IsEqual;
import org.hamcrest.core.IsNot;
import org.hamcrest.core.IsNull;
import org.hamcrest.core.IsSame;
import org.junit.runner.RunWith;
import org.mockito.Matchers;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Matchers_ESTest extends Matchers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Matcher<Float> matcher0 = IsSame.theInstance((Float) null);
      float float0 = Matchers.floatThat(matcher0);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      IsAnything<Double> isAnything0 = new IsAnything<Double>();
      Double double0 = Matchers.argThat((Matcher<Double>) isAnything0);
      assertNull(double0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Set<Object> set0 = Matchers.anySetOf(class0);
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Object object0 = Matchers.isNotNull();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      char char0 = Matchers.eq('R');
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Matcher<Boolean> matcher0 = IsSame.sameInstance((Boolean) null);
      boolean boolean0 = Matchers.booleanThat(matcher0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Double double0 = new Double(0L);
      Matcher<Double> matcher0 = Is.is(double0);
      Long long0 = new Long(0L);
      Matcher<Object> matcher1 = IsEqual.equalTo((Object) long0);
      IsNot<Object> isNot0 = new IsNot<Object>(matcher1);
      AnyOf<Double> anyOf0 = AnyOf.anyOf(matcher0, (Matcher<? super Double>) matcher0, (Matcher<? super Double>) matcher0, (Matcher<? super Double>) matcher0, (Matcher<? super Double>) isNot0, (Matcher<? super Double>) matcher1);
      double double1 = Matchers.doubleThat(anyOf0);
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      int int0 = Matchers.eq((int) (byte)0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Matcher<Integer> matcher0 = IsNot.not((Matcher<Integer>) null);
      int int0 = Matchers.intThat(matcher0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Matcher<Integer> matcher0 = IsNot.not((Matcher<Integer>) null);
      HasPropertyWithValue<Long> hasPropertyWithValue0 = new HasPropertyWithValue<Long>("org.hamcrest.core.StringContains", matcher0);
      HasPropertyWithValue<Byte> hasPropertyWithValue1 = new HasPropertyWithValue<Byte>("", hasPropertyWithValue0);
      byte byte0 = Matchers.byteThat(hasPropertyWithValue1);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Matcher<Integer> matcher0 = IsNot.not((Matcher<Integer>) null);
      HasPropertyWithValue<Long> hasPropertyWithValue0 = new HasPropertyWithValue<Long>("org.hamcrest.core.StringContains", matcher0);
      long long0 = Matchers.longThat(hasPropertyWithValue0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Object object0 = Matchers.isNull();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      char char0 = Matchers.anyChar();
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Long> class0 = Long.class;
      Long long0 = Matchers.any(class0);
      assertNull(long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String string0 = Matchers.endsWith("Y'cPE?d[.ZHK `");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Collection<Integer> collection0 = Matchers.anyCollectionOf(class0);
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = Matchers.contains("Ci");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Collection collection0 = Matchers.anyCollection();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Map map0 = Matchers.anyMap();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      String string0 = Matchers.startsWith("Ci");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = Matchers.anyString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      short short0 = Matchers.eq((short) (-1460));
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      IsArrayContaining<SelfDescribing> isArrayContaining0 = Matchers.refEq((IsArrayContaining<SelfDescribing>) null, (String[]) null);
      assertNull(isArrayContaining0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Long long0 = Matchers.anyVararg();
      assertNull(long0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Matcher<Short> matcher0 = IsNull.nullValue(class0);
      short short0 = Matchers.shortThat(matcher0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte byte0 = Matchers.eq((byte) (-1));
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      int int0 = Matchers.anyInt();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<String> class0 = String.class;
      String string0 = Matchers.isA(class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Matcher<Object>[] matcherArray0 = (Matcher<Object>[]) Array.newInstance(Matcher.class, 1);
      Matcher<Character> matcher0 = AllOf.allOf((Matcher<? super Character>[]) matcherArray0);
      char char0 = Matchers.charThat(matcher0);
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte byte0 = Matchers.anyByte();
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Integer integer0 = new Integer(0);
      Integer integer1 = Matchers.same(integer0);
      assertNull(integer1);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      String string0 = Matchers.matches("YL5uBtEFF!B[LfJsrfS");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      float float0 = Matchers.eq((-153.4448F));
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Float float0 = new Float(1791.189F);
      Float float1 = Matchers.eq(float0);
      assertNull(float1);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      long long0 = Matchers.anyLong();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double double0 = Matchers.eq(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      short short0 = Matchers.anyShort();
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      boolean boolean0 = Matchers.anyBoolean();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      float float0 = Matchers.anyFloat();
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double double0 = Matchers.anyDouble();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      long long0 = Matchers.eq((long) (byte)0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Set set0 = Matchers.anySet();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Class<Locale.Category> class0 = Locale.Category.class;
      List<Locale.Category> list0 = Matchers.anyListOf(class0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      List list0 = Matchers.anyList();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Object object0 = Matchers.any();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Matchers matchers0 = new Matchers();
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      boolean boolean0 = Matchers.eq(false);
      assertFalse(boolean0);
  }
}
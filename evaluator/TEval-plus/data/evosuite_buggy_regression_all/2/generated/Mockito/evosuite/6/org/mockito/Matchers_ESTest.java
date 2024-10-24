/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:33:52 GMT 2023
 */

package org.mockito;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.hamcrest.Matcher;
import org.hamcrest.collection.IsIn;
import org.hamcrest.core.AnyOf;
import org.hamcrest.core.CombinableMatcher;
import org.hamcrest.core.IsAnything;
import org.hamcrest.core.IsEqual;
import org.hamcrest.core.IsNull;
import org.hamcrest.number.OrderingComparison;
import org.junit.runner.RunWith;
import org.mockito.Matchers;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Matchers_ESTest extends Matchers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Float float0 = new Float((-265.848F));
      Matcher<Float> matcher0 = IsEqual.equalTo(float0);
      float float1 = Matchers.floatThat(matcher0);
      assertEquals(0.0F, float1, 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Character> class0 = Character.class;
      Matcher<Character> matcher0 = IsNull.notNullValue(class0);
      AnyOf<Character> anyOf0 = AnyOf.anyOf(matcher0, (Matcher<? super Character>) matcher0, (Matcher<? super Character>) matcher0);
      Character character0 = Matchers.argThat((Matcher<Character>) anyOf0);
      assertNull(character0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<String> class0 = String.class;
      Set<String> set0 = Matchers.anySetOf(class0);
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Object object0 = Matchers.isNotNull();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      char char0 = Matchers.eq('\'');
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Boolean> class0 = Boolean.class;
      Matcher<Boolean> matcher0 = IsNull.nullValue(class0);
      boolean boolean0 = Matchers.booleanThat(matcher0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      IsAnything<Double> isAnything0 = new IsAnything<Double>();
      double double0 = Matchers.doubleThat(isAnything0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Float> class0 = Float.class;
      Float float0 = Matchers.isNotNull(class0);
      assertNull(float0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      int int0 = Matchers.eq((-466));
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Matcher<Integer> matcher0 = IsNull.notNullValue(class0);
      int int0 = Matchers.intThat(matcher0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Matcher<Object> matcher0 = IsAnything.anything();
      CombinableMatcher<Byte> combinableMatcher0 = new CombinableMatcher<Byte>(matcher0);
      byte byte0 = Matchers.byteThat(combinableMatcher0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Matcher<Long> matcher0 = OrderingComparison.comparesEqualTo((Long) null);
      long long0 = Matchers.longThat(matcher0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Character> class0 = Character.class;
      Class<Long> class1 = Long.class;
      Map<Character, Long> map0 = Matchers.anyMapOf(class0, class1);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Object object0 = Matchers.isNull();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      char char0 = Matchers.anyChar();
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<String> class0 = String.class;
      String string0 = Matchers.any(class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = Matchers.endsWith("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Collection<Integer> collection0 = Matchers.anyCollectionOf(class0);
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = Matchers.contains("NndnZZGFdq(t^^J^");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Collection collection0 = Matchers.anyCollection();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Map map0 = Matchers.anyMap();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String string0 = Matchers.startsWith("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = Matchers.anyString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      short short0 = Matchers.eq((short)0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Integer integer0 = new Integer(0);
      String[] stringArray0 = new String[5];
      Integer integer1 = Matchers.refEq(integer0, stringArray0);
      assertNull(integer1);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Object object0 = Matchers.anyVararg();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Matcher<Short> matcher0 = IsNull.notNullValue(class0);
      short short0 = Matchers.shortThat(matcher0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte byte0 = Matchers.eq((byte) (-20));
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      int int0 = Matchers.anyInt();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Short short0 = Matchers.isA(class0);
      assertEquals((short)0, (short)short0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ArrayList<Character> arrayList0 = new ArrayList<Character>();
      Matcher<Character> matcher0 = IsIn.isIn((Collection<Character>) arrayList0);
      char char0 = Matchers.charThat(matcher0);
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte byte0 = Matchers.anyByte();
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Long long0 = new Long(0L);
      Object object0 = Matchers.same((Object) long0);
      assertTrue(object0.equals((Object)long0));
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      String string0 = Matchers.matches("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      float float0 = Matchers.eq((-5729.0F));
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Boolean boolean0 = Boolean.valueOf(false);
      Boolean boolean1 = Matchers.eq(boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      long long0 = Matchers.anyLong();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double double0 = Matchers.eq((-1467.919));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      short short0 = Matchers.anyShort();
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      boolean boolean0 = Matchers.anyBoolean();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      float float0 = Matchers.anyFloat();
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double double0 = Matchers.anyDouble();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      long long0 = Matchers.eq((-476L));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Set set0 = Matchers.anySet();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Class<String> class0 = String.class;
      List<String> list0 = Matchers.anyListOf(class0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      List list0 = Matchers.anyList();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Byte byte0 = Matchers.any();
      assertNull(byte0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Short short0 = Matchers.isNull(class0);
      assertNull(short0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Matchers matchers0 = new Matchers();
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      boolean boolean0 = Matchers.eq(false);
      assertFalse(boolean0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:16:01 GMT 2023
 */

package org.mockito;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.hamcrest.Matcher;
import org.hamcrest.core.Is;
import org.hamcrest.core.IsAnything;
import org.hamcrest.core.IsEqual;
import org.hamcrest.core.IsInstanceOf;
import org.hamcrest.core.IsNull;
import org.hamcrest.core.IsSame;
import org.hamcrest.number.OrderingComparison;
import org.hamcrest.object.HasToString;
import org.junit.runner.RunWith;
import org.mockito.Matchers;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Matchers_ESTest extends Matchers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Float float0 = new Float((-296.58078F));
      IsEqual<Float> isEqual0 = new IsEqual<Float>(float0);
      float float1 = Matchers.floatThat(isEqual0);
      assertEquals(0.0F, float1, 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      IsNull<Double> isNull0 = new IsNull<Double>();
      Double double0 = Matchers.argThat((Matcher<Double>) isNull0);
      assertNull(double0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Object object0 = Matchers.isNotNull();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Set<Object> set0 = Matchers.anySetOf(class0);
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      char char0 = Matchers.eq('D');
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      IsAnything<Boolean> isAnything0 = new IsAnything<Boolean>();
      boolean boolean0 = Matchers.booleanThat(isAnything0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Double double0 = new Double((-1002.2));
      Matcher<Double> matcher0 = OrderingComparison.comparesEqualTo(double0);
      double double1 = Matchers.doubleThat(matcher0);
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      String string0 = Matchers.isNotNull(class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      int int0 = Matchers.eq((int) (short)1652);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Matcher<Integer> matcher0 = Is.is((Matcher<Integer>) null);
      int int0 = Matchers.intThat(matcher0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Matcher<Object> matcher0 = IsInstanceOf.any(class0);
      HasToString<Byte> hasToString0 = new HasToString<Byte>(matcher0);
      byte byte0 = Matchers.byteThat(hasToString0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      long long0 = Matchers.longThat((Matcher<Long>) null);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<String> class0 = String.class;
      Map<String, String> map0 = Matchers.anyMapOf(class0, class0);
      assertTrue(map0.isEmpty());
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
      Class<Object> class0 = Object.class;
      Object object0 = Matchers.any(class0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = Matchers.endsWith("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Float> class0 = Float.class;
      Collection<Float> collection0 = Matchers.anyCollectionOf(class0);
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = Matchers.contains((String) null);
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
      String string0 = Matchers.startsWith("ki?*[");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = Matchers.anyString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      short short0 = Matchers.eq((short)1652);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      String[] stringArray0 = new String[0];
      Integer integer0 = Matchers.refEq((Integer) null, stringArray0);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Short short0 = Matchers.anyVararg();
      assertNull(short0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Short short0 = new Short((short)0);
      Matcher<Short> matcher0 = IsSame.theInstance(short0);
      short short1 = Matchers.shortThat(matcher0);
      assertEquals((short)0, short1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Boolean boolean0 = Matchers.any();
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte byte0 = Matchers.eq((byte) (-128));
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      int int0 = Matchers.anyInt();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Short short0 = Matchers.isA(class0);
      assertEquals((short)0, (short)short0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Character character0 = Character.valueOf('8');
      Matcher<Character> matcher0 = OrderingComparison.lessThan(character0);
      char char0 = Matchers.charThat(matcher0);
      assertEquals('\u0000', char0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte byte0 = Matchers.anyByte();
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Float float0 = new Float(0.0F);
      Float float1 = Matchers.same(float0);
      assertEquals((float)float1, (float)float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      String string0 = Matchers.matches("EzMFs4Jdid");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      float float0 = Matchers.eq(0.0F);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Character character0 = Character.valueOf('\u0000');
      Character character1 = Matchers.eq(character0);
      assertEquals('\u0000', (char)character1);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      long long0 = Matchers.anyLong();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double double0 = Matchers.eq(2728.0059);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      short short0 = Matchers.anyShort();
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      boolean boolean0 = Matchers.anyBoolean();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      float float0 = Matchers.anyFloat();
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double double0 = Matchers.anyDouble();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      long long0 = Matchers.eq(3508L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Set set0 = Matchers.anySet();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Class<Long> class0 = Long.class;
      List<Long> list0 = Matchers.anyListOf(class0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      List list0 = Matchers.anyList();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Class<Long> class0 = Long.class;
      Long long0 = Matchers.isNull(class0);
      assertNull(long0);
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

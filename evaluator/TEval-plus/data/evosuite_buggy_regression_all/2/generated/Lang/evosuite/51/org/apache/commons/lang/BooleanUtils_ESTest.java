/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:22:47 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.lang.BooleanUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BooleanUtils_ESTest extends BooleanUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject(0);
      String string0 = BooleanUtils.toStringYesNo(boolean0);
      assertNotNull(string0);
      assertEquals("no", string0);
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      String string0 = BooleanUtils.toStringYesNo(false);
      assertEquals("no", string0);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      BooleanUtils booleanUtils0 = new BooleanUtils();
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      String string0 = BooleanUtils.toStringTrueFalse(false);
      assertEquals("false", string0);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      String string0 = BooleanUtils.toStringTrueFalse((Boolean) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      String string0 = BooleanUtils.toStringOnOff(true);
      assertEquals("on", string0);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      Boolean boolean0 = Boolean.valueOf("9");
      String string0 = BooleanUtils.toStringOnOff(boolean0);
      assertNotNull(string0);
      assertEquals("off", string0);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.negate((Boolean) false);
      assertTrue(boolean0);
      assertNotNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.negate((Boolean) null);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.negate((Boolean) true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      boolean boolean0 = BooleanUtils.isNotTrue((Boolean) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      Boolean boolean0 = Boolean.FALSE;
      boolean boolean1 = BooleanUtils.isTrue(boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      Boolean boolean0 = new Boolean(true);
      boolean boolean1 = BooleanUtils.isNotTrue(boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      Boolean boolean0 = Boolean.FALSE;
      boolean boolean1 = BooleanUtils.isNotFalse(boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      boolean boolean0 = BooleanUtils.isNotFalse((Boolean) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((String) null, (String) null, (String) null, "2*7lN!0");
      boolean boolean1 = BooleanUtils.isNotFalse(boolean0);
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject(false);
      Integer integer0 = BooleanUtils.toIntegerObject(boolean0);
      assertEquals(0, (int)integer0);
      assertNotNull(integer0);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject(true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      Boolean boolean0 = new Boolean("K:N");
      boolean boolean1 = BooleanUtils.toBoolean(boolean0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean((Boolean) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      boolean boolean1 = BooleanUtils.toBoolean(boolean0);
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      Boolean boolean0 = new Boolean(", '");
      boolean boolean1 = BooleanUtils.toBooleanDefaultIfNull(boolean0, false);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBooleanDefaultIfNull((Boolean) null, false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      boolean boolean1 = BooleanUtils.toBooleanDefaultIfNull(boolean0, false);
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean(1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean(0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((-383));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      Integer integer0 = new Integer(45);
      Boolean boolean0 = BooleanUtils.toBooleanObject(integer0);
      Integer integer1 = BooleanUtils.toIntegerObject(boolean0, integer0, integer0, integer0);
      assertEquals(45, (int)integer1);
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((Integer) null);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      Integer integer0 = BooleanUtils.toIntegerObject(false);
      Boolean boolean0 = BooleanUtils.toBooleanObject(integer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean(582, (-5002), 582);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean(13, 13, 13);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean(97, 83, 83);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Integer did not match either specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      Integer integer0 = new Integer((-28));
      boolean boolean0 = BooleanUtils.toBoolean(integer0, integer0, integer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean((Integer) null, (Integer) null, (Integer) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      Integer integer0 = BooleanUtils.toIntegerObject(false);
      assertEquals(0, (int)integer0);
      
      boolean boolean0 = BooleanUtils.toBoolean((Integer) null, integer0, (Integer) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      Integer integer0 = BooleanUtils.toIntegerObject(false);
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean((Integer) null, integer0, integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Integer did not match either specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      Integer integer0 = new Integer(121);
      Integer integer1 = new Integer((-3406));
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean(integer0, integer1, integer1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Integer did not match either specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      Integer integer0 = new Integer((-623));
      Integer integer1 = new Integer(89);
      boolean boolean0 = BooleanUtils.toBoolean(integer0, integer1, integer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((-1875), 0, 0, (-1875));
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject(48, 56, 48, 56);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBooleanObject(2072, Integer.MAX_VALUE, 240, 240);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Integer did not match any specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((Integer) null, (Integer) null, (Integer) null, (Integer) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      Integer integer0 = new Integer(45);
      // Undeclared exception!
      try { 
        BooleanUtils.toBooleanObject((Integer) null, integer0, integer0, integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Integer did not match any specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      Integer integer0 = Integer.getInteger("");
      Integer integer1 = new Integer(43);
      Boolean boolean0 = BooleanUtils.toBooleanObject((Integer) null, integer1, (Integer) null, integer0);
      assertFalse(boolean0);
      assertNotNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      Integer integer0 = new Integer(117);
      Boolean boolean0 = BooleanUtils.toBooleanObject((Integer) null, integer0, integer0, (Integer) null);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      Integer integer0 = new Integer(111);
      Integer integer1 = new Integer((-2028178998));
      Boolean boolean0 = BooleanUtils.toBooleanObject(integer0, integer1, integer0, integer1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      Integer integer0 = new Integer(117);
      Integer integer1 = new Integer(46);
      Boolean boolean0 = BooleanUtils.toBooleanObject(integer1, integer0, integer0, integer1);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      Integer integer0 = new Integer(2460);
      Integer integer1 = new Integer((-2719));
      // Undeclared exception!
      try { 
        BooleanUtils.toBooleanObject(integer1, integer0, integer0, integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Integer did not match any specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      int int0 = BooleanUtils.toInteger(false);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      int int0 = BooleanUtils.toInteger(true);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      Integer integer0 = BooleanUtils.toIntegerObject(true);
      assertEquals(1, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject(2223, 2223, 2223, 2223);
      Integer integer0 = BooleanUtils.toIntegerObject(boolean0);
      assertNotNull(integer0);
      assertEquals(1, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      Integer integer0 = BooleanUtils.toIntegerObject((Boolean) null);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      int int0 = BooleanUtils.toInteger(false, (-623), (-623));
      assertEquals((-623), int0);
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      int int0 = BooleanUtils.toInteger(true, (-623), 89);
      assertEquals((-623), int0);
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      Boolean boolean0 = Boolean.valueOf(false);
      int int0 = BooleanUtils.toInteger(boolean0, 0, 0, (-1277));
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      int int0 = BooleanUtils.toInteger((Boolean) null, 395, 395, 395);
      assertEquals(395, int0);
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      Boolean boolean0 = Boolean.valueOf(true);
      int int0 = BooleanUtils.toInteger(boolean0, 0, 0, (-1277));
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      Integer integer0 = new Integer((-3321));
      Integer integer1 = BooleanUtils.toIntegerObject(false, integer0, integer0);
      assertEquals((-3321), (int)integer1);
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      Integer integer0 = new Integer(0);
      Integer integer1 = BooleanUtils.toIntegerObject(true, integer0, integer0);
      assertEquals(0, (int)integer1);
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      Integer integer0 = new Integer(2223);
      Integer integer1 = BooleanUtils.toIntegerObject((Boolean) null, integer0, integer0, integer0);
      assertEquals(2223, (int)integer1);
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      Boolean boolean0 = Boolean.FALSE;
      Integer integer0 = new Integer(114);
      Integer integer1 = BooleanUtils.toIntegerObject(boolean0, integer0, integer0, integer0);
      assertEquals(114, (int)integer1);
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("off");
      assertFalse(boolean0);
      assertNotNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("true");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("false");
      assertNotNull(boolean0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("on");
      assertNotNull(boolean0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("~-");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("yes");
      assertNotNull(boolean0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("no");
      assertFalse(boolean0);
      assertNotNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("2*7lN!0", (String) null, (String) null, "2*7lN!0");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((String) null, "trP", (String) null, "trP");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject((String) null, "2*7lN!0", "P55$55K%", (String) null);
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBooleanObject((String) null, "Tl|`", "~,A>?XaQW9", "~,A>?XaQW9");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The String did not match any specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      Boolean boolean0 = BooleanUtils.toBooleanObject("K*7lN!0", "K*7lN!0", "K*7lN!0", "K*7lN!0");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBooleanObject("yeI", "[|n0L#M\"B!Eo", "[|n0L#M\"B!Eo", ",#_BIfp:h=/");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The String did not match any specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("no");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("true");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("yes");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("2*7lN!0");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("o ");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("ON");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("on");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("K:N");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("yof");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("yEl");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("Yes");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("YEL");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("Ym$");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean("tRu");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("tj+u");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("trE");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean("trU");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("trus");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("TRts");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("TrUs");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("Ti-X");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("", "true", "");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean((String) null, "tr[P", (String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean((String) null, (String) null, (String) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean((String) null, "'kt'e", "o ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The String did not match either specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      boolean boolean0 = BooleanUtils.toBoolean("', haN a kength less than 2", "', haN a kength less than 2", "', haN a kength less than 2");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.toBoolean("The Integer did not match either specified value", (String) null, (String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The String did not match either specified value
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      Integer integer0 = new Integer((-2126));
      Boolean boolean0 = BooleanUtils.toBooleanObject(integer0, integer0, integer0, integer0);
      String string0 = BooleanUtils.toStringYesNo(boolean0);
      assertNotNull(string0);
      assertEquals("yes", string0);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      boolean[] booleanArray0 = new boolean[1];
      boolean boolean0 = BooleanUtils.xor(booleanArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.xor((boolean[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      boolean[] booleanArray0 = new boolean[0];
      // Undeclared exception!
      try { 
        BooleanUtils.xor(booleanArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array is empty
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      Integer integer0 = new Integer(45);
      Boolean boolean0 = BooleanUtils.toBooleanObject(integer0);
      assertTrue(boolean0);
      
      Boolean[] booleanArray0 = new Boolean[2];
      booleanArray0[0] = boolean0;
      booleanArray0[1] = boolean0;
      Boolean boolean1 = BooleanUtils.xor(booleanArray0);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      Boolean[] booleanArray0 = new Boolean[2];
      Boolean boolean0 = BooleanUtils.toBooleanObject(1937, 1937, (-2656), 1937);
      assertNotNull(boolean0);
      
      booleanArray0[0] = boolean0;
      Boolean boolean1 = BooleanUtils.toBooleanObject("", "BU0zWH<QCWJi0", "", "BU0zWH<QCWJi0");
      booleanArray0[1] = boolean1;
      Boolean boolean2 = BooleanUtils.xor(booleanArray0);
      assertTrue(boolean2);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      // Undeclared exception!
      try { 
        BooleanUtils.xor((Boolean[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      Boolean[] booleanArray0 = new Boolean[0];
      // Undeclared exception!
      try { 
        BooleanUtils.xor(booleanArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array is empty
         //
         verifyException("org.apache.commons.lang.BooleanUtils", e);
      }
  }
}
